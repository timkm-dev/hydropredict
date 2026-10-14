"""Evaluate frozen raw/temporal models on reconstructed validation context only."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from compare_validation_models import event_metrics, nominal_metrics

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.context import reconstruct_context
from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import PartitionManifest, compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.models.isolation_forest import SHPIsolationForest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", required=True, type=Path)
    parser.add_argument("--comparison-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory to preserve prior evidence.")
    root = Path(__file__).resolve().parents[1]
    manifest = PartitionManifest.load(args.preparation_run / "partitions/partitions_manifest.json")
    for source, checksum in manifest.source_checksums.items():
        if compute_file_sha256(source) != checksum:
            raise ValueError(f"Source changed: {source}")
    lower, upper = pd.Timestamp("2019-02-01", tz="UTC"), pd.Timestamp("2019-05-01", tz="UTC")
    normal = load_telemetry_dataframe(root / "data/raw/shp_train.csv", include_source_row=True)
    prefault = load_telemetry_dataframe(root / "data/raw/shp_test.csv", include_source_row=True)
    timeline, provenance, reconstruction = reconstruct_context(
        {"published_normal": normal, "published_prefault": prefault},
        lower,
        upper,
    )
    normal = normal[(normal.t >= lower) & (normal.t < upper)].copy()
    prefault = prefault[(prefault.t >= lower) & (prefault.t < upper)].copy()
    faults = [f for f in load_fault_log(root / "data/raw/shp_faults.csv") if lower <= f < upper]
    original_coverage = fault_window_coverage(prefault.t, faults)
    joined_coverage = fault_window_coverage(timeline.t, faults)
    args.output.mkdir(parents=True)
    protocol = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "interval": [lower.isoformat(), upper.isoformat()],
        "models": ["raw-q0.99", "temporal-q0.99"],
        "retraining": False,
        "threshold_changes": False,
        "test_period_scored": False,
        "source_checksums": manifest.source_checksums,
        "policy": "Exact timestamp union; no fill; original coverage and scored coverage separate",
        "reconstruction": reconstruction,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    timeline.to_csv(args.output / "validation_timeline.csv", index=False)
    provenance.to_csv(args.output / "source_provenance.csv", index=False)
    results = []
    for mode in ("raw", "temporal"):
        model = SHPIsolationForest.load(args.comparison_run / f"{mode}-q0.99")
        previous = model.score(prefault)
        joined = model.score(timeline)
        common = joined[joined.t.isin(previous.t)]
        # Raw predictions must be exactly unaffected by additional context.
        if mode == "raw":
            np.testing.assert_array_equal(previous.anomaly_score, common.anomaly_score)
        # Temporal histories may change legitimately, but not after their output time.
        boundary = lower + pd.Timedelta(days=2)
        prefix = model.score(timeline[timeline.t < boundary])
        np.testing.assert_allclose(
            prefix.anomaly_score, joined[joined.t.isin(prefix.t)].anomaly_score
        )
        prior_normal = model.score(normal)
        joined_normal = joined[joined.t.isin(normal.t)]
        common_normal = joined_normal[joined_normal.t.isin(prior_normal.t)]
        events = event_metrics(joined, original_coverage)
        scored_coverage = fault_window_coverage(joined.t, faults)
        eligible_times = {e["fault_time"] for e in scored_coverage if e["eligible"]}
        result = {
            "mode": mode,
            "model_version": model.version,
            "threshold": model.threshold,
            "timeline_rows": len(timeline),
            "scored_rows": len(joined),
            "prefault_score_rows_before": len(previous),
            "prefault_score_rows_after": int(joined.t.isin(prefault.t).sum()),
            "original_coverage": original_coverage,
            "joined_observation_coverage": joined_coverage,
            "scored_coverage": scored_coverage,
            "original_eligible_event_results": events,
            "model_eligible_events": len(eligible_times),
            "model_eligible_detections": sum(
                e["detected"] and e["fault_time"] in eligible_times for e in events["events"]
            ),
            "nominal_all_scored": nominal_metrics(joined_normal),
            "nominal_matched_before": nominal_metrics(prior_normal),
            "nominal_matched_after": nominal_metrics(common_normal),
        }
        results.append(result)
        joined.to_csv(args.output / f"{mode}_scores.csv", index=False)
        print(
            mode,
            "prefault rows",
            result["prefault_score_rows_before"],
            "->",
            result["prefault_score_rows_after"],
            "adequate events",
            result["model_eligible_events"],
            "detected",
            result["model_eligible_detections"],
            "nominal alerts",
            result["nominal_all_scored"]["point_alert_fraction"],
        )
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    hashes = {p.name: compute_file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
