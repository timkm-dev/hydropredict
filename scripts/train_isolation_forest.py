"""Train and evaluate a frozen exploratory Isolation Forest without test-set tuning."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import PartitionManifest, compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.models.isolation_forest import SHPIsolationForest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error("Choose a new output directory; existing runs are preserved.")
    manifest = PartitionManifest.load(args.preparation_run / "partitions/partitions_manifest.json")
    if not manifest.is_strictly_chronological:
        raise ValueError("Baseline requires chronological partitions.")
    frames = {}
    for name, part in manifest.partitions.items():
        if compute_file_sha256(part.relative_path) != part.sha256:
            raise ValueError(f"Partition changed: {name}")
        frames[name] = load_telemetry_dataframe(part.relative_path)
    normal = frames["train"]
    split = int(len(normal) * 0.8)
    fitting, calibration = normal.iloc[:split].copy(), normal.iloc[split:].copy()
    if not (
        calibration.t.max() < frames["validation"].t.min()
        and frames["validation"].t.max() < frames["test"].t.min()
    ):
        raise ValueError("Fitting, calibration, validation and testing must be ordered.")
    model = SHPIsolationForest()
    start = perf_counter()
    model.fit(fitting, calibration, version=args.output.name)
    elapsed = perf_counter() - start
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    provenance = {
        "source_commit": commit,
        "python_version": platform.python_version(),
        "training_source_hash": manifest.partitions["train"].sha256,
        "fit_rows": len(fitting),
        "calibration_rows": len(calibration),
        "fit_end": fitting.t.max().isoformat(),
        "calibration_start": calibration.t.min().isoformat(),
        "calibration_end": calibration.t.max().isoformat(),
        "calibration_quantile": 0.99,
        "training_seconds": elapsed,
        "status": "exploratory baseline; publisher-selected reference data",
    }
    model.save(args.output, provenance)
    reloaded = SHPIsolationForest.load(args.output)
    reference = model.score(calibration)
    np.testing.assert_allclose(reference.anomaly_score, reloaded.score(calibration).anomaly_score)
    reference.to_csv(args.output / "calibration_scores.csv", index=False)
    # Evaluation begins only after model and threshold are frozen and exported.
    faults = load_fault_log(root / "data/raw/shp_faults.csv")
    evaluation: dict[str, object] = {"training": provenance, "threshold": model.threshold}
    for name in ("validation", "test"):
        frame = frames[name]
        scores = reloaded.score(frame)
        scores.to_csv(args.output / f"{name}_scores.csv", index=False)
        lower = pd.Timestamp("2019-02-01" if name == "validation" else "2019-05-01", tz="UTC")
        upper = pd.Timestamp("2019-05-01" if name == "validation" else "2019-08-10", tz="UTC")
        events = [f for f in faults if lower <= f < upper]
        coverage = fault_window_coverage(frame.t, events)
        alert_times = pd.to_datetime(scores.loc[scores.is_alert, "t"], utc=True)
        lead_times = []
        for event in coverage:
            fault = pd.Timestamp(event["fault_time"])
            candidates = alert_times[
                (alert_times < fault) & (alert_times >= fault - pd.Timedelta(hours=6))
            ]
            detected = bool(len(candidates))
            event["detected"] = detected
            event["lead_time_seconds"] = (
                float((fault - candidates.min()).total_seconds()) if detected else None
            )
            if event["eligible"] and detected:
                lead_times.append(event["lead_time_seconds"])
        eligible = sum(e["eligible"] for e in coverage)
        detected_count = sum(e["eligible"] and e["detected"] for e in coverage)
        evaluation[name] = {
            "candidate_events": len(events),
            "eligible_events": eligible,
            "detected_eligible_events": detected_count,
            "eligible_event_recall": detected_count / eligible if eligible else None,
            "detected_any_coverage_events": sum(e["detected"] for e in coverage),
            "median_eligible_lead_seconds": float(np.median(lead_times)) if lead_times else None,
            "alert_observations": int(scores.is_alert.sum()),
            "observations": len(scores),
            "events": coverage,
            "false_alert_frequency": None,
            "false_alert_reason": "Fault-selected excerpts do not represent normal exposure.",
        }
    # Separate, later publisher-selected normal data provides a biased nominal reference only.
    healthy = load_telemetry_dataframe(root / "data/raw/shp_train.csv")
    held_normal = healthy[healthy.t >= pd.Timestamp("2019-05-01", tz="UTC")]
    nominal = reloaded.score(held_normal)
    nominal.to_csv(args.output / "held_normal_scores.csv", index=False)
    evaluation["later_nominal_reference"] = {
        "observations": len(nominal),
        "alert_observations": int(nominal.is_alert.sum()),
        "point_exceedance_fraction": float(nominal.is_alert.mean()),
        "interpretation": "Selected normal reference, not fleet-wide false-alert frequency.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    files = {p.name: compute_file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: ({a: b for a, b in v.items() if a != "events"} if isinstance(v, dict) else v)
                for k, v in evaluation.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
