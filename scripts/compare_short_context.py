"""One fixed 15/30/120-minute validation comparison, without final-test evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from compare_validation_models import event_metrics, nominal_metrics

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import PartitionManifest, compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.features.causal import causal_feature_frame
from hydropredict.models.isolation_forest import SHPIsolationForest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", type=Path, required=True)
    parser.add_argument("--context-run", type=Path, required=True)
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; preserve previous experiments.")
    root = Path(__file__).resolve().parents[1]
    manifest = PartitionManifest.load(args.preparation_run / "partitions/partitions_manifest.json")
    if not manifest.is_strictly_chronological:
        raise ValueError("Chronological source partitions required.")
    checksums = json.loads((args.context_run / "checksums.json").read_text(encoding="utf-8"))
    for name in ("validation_timeline.csv", "source_provenance.csv", "protocol.json"):
        if compute_file_sha256(args.context_run / name) != checksums[name]:
            raise ValueError(f"Context artifact changed: {name}")
    for path, checksum in manifest.source_checksums.items():
        if compute_file_sha256(path) != checksum:
            raise ValueError(f"Source changed: {path}")
    part = manifest.partitions["train"]
    if compute_file_sha256(part.relative_path) != part.sha256:
        raise ValueError("Training partition changed.")
    training = load_telemetry_dataframe(part.relative_path)
    split = int(len(training) * 0.8)
    fitting, calibration = training.iloc[:split], training.iloc[split:]
    timeline = load_telemetry_dataframe(args.context_run / "validation_timeline.csv")
    lower, upper = pd.Timestamp("2019-02-01", tz="UTC"), pd.Timestamp("2019-05-01", tz="UTC")
    if calibration.t.max() >= lower or not timeline.t.between(lower, upper, inclusive="left").all():
        raise ValueError("Validation interval or calibration boundary changed.")
    normal = load_telemetry_dataframe(root / "data/raw/shp_train.csv")
    nominal_times = normal.loc[(normal.t >= lower) & (normal.t < upper), "t"]
    faults = [f for f in load_fault_log(root / "data/raw/shp_faults.csv") if lower <= f < upper]
    validation_part = manifest.partitions["validation"]
    if compute_file_sha256(validation_part.relative_path) != validation_part.sha256:
        raise ValueError("Validation partition changed.")
    prefault = load_telemetry_dataframe(validation_part.relative_path)
    original_coverage = fault_window_coverage(prefault.t, faults)
    common_times = causal_feature_frame(timeline, "temporal", 120).t
    args.output.mkdir(parents=True)
    protocol = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "contexts_minutes": [15, 30, 120],
        "calibration_quantile": 0.99,
        "windows_minutes": {"15": [10, 15], "30": [15, 30], "120": [30, 120]},
        "features": "36: six raw channels, two rolling mean/std sets, six first derivatives",
        "gap_reset_seconds": 450,
        "interpolation": False,
        "warmup": "Full longest context since segment start; no cross-boundary history",
        "fit_rows_before_warmup": len(fitting),
        "calibration_rows_before_warmup": len(calibration),
        "training_hash": part.sha256,
        "source_checksums": manifest.source_checksums,
        "coverage_rule": "Unchanged six-hour/80%/last-observation-within-ten-minutes policy",
        "development_budget": "2% point alerts on selected nominal reference; not a guarantee",
        "test_evaluated": False,
        "model_promoted": False,
        "stop_rule": "Run these three configurations once; no further Isolation Forest search",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for minutes in (15, 30, 120):
        start = perf_counter()
        if minutes == 120:
            model = SHPIsolationForest.load(args.comparison_run / "temporal-q0.99")
            training_seconds = None
        else:
            model = SHPIsolationForest(feature_mode="temporal", context_minutes=minutes)
            model.fit(fitting, calibration, version=f"{args.output.name}-{minutes}min")
            training_seconds = perf_counter() - start
            model.save(args.output / f"model-{minutes}min", protocol)
            restored = SHPIsolationForest.load(args.output / f"model-{minutes}min")
            np.testing.assert_allclose(
                model.score(timeline).anomaly_score, restored.score(timeline).anomaly_score
            )
            model = restored
        scores = model.score(timeline)
        prefix = model.score(timeline[timeline.t < lower + pd.Timedelta(days=2)])
        np.testing.assert_allclose(
            prefix.anomaly_score, scores[scores.t.isin(prefix.t)].anomaly_score
        )
        scored_coverage = fault_window_coverage(scores.t, faults)
        nominal = scores[scores.t.isin(nominal_times)]
        paired = scores[scores.t.isin(common_times)]
        paired_nominal = paired[paired.t.isin(nominal_times)]
        result = {
            "context_minutes": minutes,
            "model_version": model.version,
            "threshold": model.threshold,
            "training_seconds": training_seconds,
            "fit_rows_after_warmup": len(causal_feature_frame(fitting, "temporal", minutes)),
            "calibration_rows_after_warmup": len(
                causal_feature_frame(calibration, "temporal", minutes)
            ),
            "timeline_rows": len(timeline),
            "scored_rows": len(scores),
            "original_eligible": event_metrics(scores, original_coverage),
            "adequately_scored": event_metrics(scores, scored_coverage),
            "paired_original_eligible": event_metrics(paired, original_coverage),
            "nominal": nominal_metrics(nominal),
            "paired_nominal": nominal_metrics(paired_nominal),
        }
        results.append(result)
        scores.to_csv(args.output / f"scores-{minutes}min.csv", index=False)
        print(
            minutes,
            "minutes; scored",
            len(scores),
            "effective eligible/detected",
            result["adequately_scored"]["eligible_events"],
            result["adequately_scored"]["detected_events"],
            "nominal fraction",
            result["nominal"]["point_alert_fraction"],
        )
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    hashes = {p.name: compute_file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
