"""Audit source integrity, chronological partitions and pre-fault observation coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import PartitionManifest, compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = PartitionManifest.load(args.preparation_run / "partitions/partitions_manifest.json")
    if not manifest.is_strictly_chronological:
        raise ValueError("Readiness requires chronological partitions.")
    for path, checksum in manifest.source_checksums.items():
        if compute_file_sha256(path) != checksum:
            raise ValueError(f"Source changed: {path}")
    faults = load_fault_log(root / "data/raw/shp_faults.csv")
    cutoffs = [pd.Timestamp("2019-02-01", tz="UTC"), pd.Timestamp("2019-05-01", tz="UTC")]
    report: dict[str, object] = {
        "decision": "Conditional GO for an exploratory six-channel Isolation Forest baseline",
        "source_hashes_verified": True,
        "total_faults": len(faults),
        "coverage_policy": {
            "warning_hours": 6,
            "minimum_fraction": 0.8,
            "maximum_last_gap_seconds": 600,
            "cadence_seconds": 300,
        },
        "limitations": [
            "Source training is a publisher-selected normal reference, not certified health.",
            "The published test split uses fault-selected 12-hour excerpts.",
            "No new fault-derived filtering is allowed during training or threshold calibration.",
            "Channel units/order and timezone remain unresolved; do not diagnose physically.",
            "Fault-adjacent excerpts cannot estimate a fleet-wide false-alert frequency.",
        ],
    }
    frames = {}
    for name, metadata in manifest.partitions.items():
        if compute_file_sha256(metadata.relative_path) != metadata.sha256:
            raise ValueError(f"Partition changed: {name}")
        frames[name] = load_telemetry_dataframe(metadata.relative_path)
    if not (
        frames["train"].t.max()
        < frames["validation"].t.min()
        <= frames["validation"].t.max()
        < frames["test"].t.min()
    ):
        raise ValueError("Partition temporal ordering failed.")
    report["partition_rows"] = {name: len(frame) for name, frame in frames.items()}
    for name, lower, upper in [
        ("validation", cutoffs[0], cutoffs[1]),
        ("test", cutoffs[1], pd.Timestamp("2019-08-10", tz="UTC")),
    ]:
        events = [f for f in faults if lower <= f < upper]
        coverage = fault_window_coverage(frames[name].t, events)
        report[name] = {
            "candidate_events": len(events),
            "events_with_any_prior_data": sum(r["has_any_prior_data"] for r in coverage),
            "eligible_events": sum(r["eligible"] for r in coverage),
            "event_coverage": coverage,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("validation", "test")}, indent=2)
    )
    for name in ("validation", "test"):
        section = report[name]
        assert isinstance(section, dict)
        print(name, {k: v for k, v in section.items() if k != "event_coverage"})


if __name__ == "__main__":
    main()
