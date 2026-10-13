"""Create one reproducible SHP preparation run without overwriting prior artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hydropredict.data.partitioning import compute_file_sha256, create_shp_partitions
from hydropredict.data.preprocessing import (
    StandardTelemetryScaler,
    load_telemetry_dataframe,
    prepare_telemetry_series,
)
from hydropredict.data.schema import MEASUREMENT_COLUMNS
from hydropredict.features.pipeline import extract_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        parser.error("Run directory already exists; choose a new name to preserve prior artifacts.")
    raw_dir = root / "data" / "raw"
    manifest = create_shp_partitions(
        raw_dir / "shp_train.csv",
        raw_dir / "shp_test.csv",
        raw_dir / "shp_faults.csv",
        run_dir / "partitions",
    )
    scaler = StandardTelemetryScaler()
    summary: dict[str, object] = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "partition_manifest": "partitions/partitions_manifest.json",
        "preparation": {"freq": "5min", "method": "forward", "max_fill_steps": 3},
        "scaling": "standard; fit on prepared training partition only",
        "limitations": [
            "Offline segment preparation; live replay causality still needs integration tests.",
            "Chronological cutoffs exclude source rows; see partition source_accounting.",
            "Model training, fault-event coverage and warning performance are not evaluated here.",
        ],
    }
    artifacts: dict[str, str] = {}
    for name, metadata in manifest.partitions.items():
        source = Path(metadata.relative_path)
        if compute_file_sha256(source) != metadata.sha256:
            raise ValueError(f"Partition checksum mismatch: {name}")
        prepared, accounting = prepare_telemetry_series(load_telemetry_dataframe(source))
        if name == "train":
            scaler.fit(prepared)
            scaler.save(run_dir / "scaler.json")
        scaled = scaler.transform(prepared)
        # Physical ratios are calculated before scaling, in original measurement units.
        features = extract_feature_matrix(prepared)
        for stage, frame in (("prepared", prepared), ("scaled", scaled), ("features", features)):
            if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
                raise ValueError(f"Nonfinite values in {name}/{stage}")
            filename = f"shp_{name}_{stage}.csv"
            frame.to_csv(run_dir / filename, index=False)
            artifacts[filename] = compute_file_sha256(run_dir / filename)
        summary[name] = {
            **accounting.summary(),
            "feature_columns": len(features.columns) - 2,  # timestamp and segment coordinate
            "measurement_columns": list(MEASUREMENT_COLUMNS),
        }
    artifacts["scaler.json"] = compute_file_sha256(run_dir / "scaler.json")
    summary["artifacts_sha256"] = artifacts
    (run_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
