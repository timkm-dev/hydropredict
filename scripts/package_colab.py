"""Create one portable ZIP from committed code and explicitly selected SHP inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import PartitionManifest, compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", required=True, type=Path)
    parser.add_argument("--context-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; prior packages are preserved.")
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit source changes before packaging a reproducible snapshot.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=zip",
            "HEAD",
            "pyproject.toml",
            "README.md",
            "src",
            "scripts",
            "configs",
            "notebooks",
        ],
        cwd=root,
    )
    manifest = PartitionManifest.load(args.preparation_run / "partitions/partitions_manifest.json")
    if not manifest.is_strictly_chronological:
        raise ValueError("Chronological preparation required.")
    for path, checksum in manifest.source_checksums.items():
        if compute_file_sha256(path) != checksum:
            raise ValueError(f"Source changed: {path}")
    context_hashes = json.loads((args.context_run / "checksums.json").read_text(encoding="utf-8"))
    timeline_path = args.context_run / "validation_timeline.csv"
    if compute_file_sha256(timeline_path) != context_hashes["validation_timeline.csv"]:
        raise ValueError("Validation context changed.")
    training = manifest.partitions["train"]
    validation = manifest.partitions["validation"]
    for part in (training, validation):
        if compute_file_sha256(part.relative_path) != part.sha256:
            raise ValueError(f"Partition changed: {part.partition_name}")
    normal = load_telemetry_dataframe(root / "data/raw/shp_train.csv")

    lower, upper = pd.Timestamp("2019-02-01", tz="UTC"), pd.Timestamp("2019-05-01", tz="UTC")
    nominal = normal[(normal.t >= lower) & (normal.t < upper)]
    payloads = {
        "project.zip": archive,
        "data/normal_training.csv": Path(training.relative_path).read_bytes(),
        "data/validation_context.csv": timeline_path.read_bytes(),
        "data/validation_prefault.csv": Path(validation.relative_path).read_bytes(),
        "data/validation_nominal.csv": nominal.to_csv(index=False).encode(),
        # This file is only consumed by the separate evaluation command, never training.
        "data/evaluation_faults.csv": pd.DataFrame(
            {
                "t": [
                    t.isoformat()
                    for t in load_fault_log(root / "data/raw/shp_faults.csv")
                    if lower <= t < upper
                ]
            }
        )
        .to_csv(index=False)
        .encode(),
    }

    hashes = {name: hashlib.sha256(data).hexdigest().upper() for name, data in payloads.items()}
    input_manifest = {
        "source_commit": commit,
        "files_sha256": hashes,
        "validation_interval": [lower.isoformat(), upper.isoformat()],
        "training_role": "Publisher-selected normal reference; inherits upstream fault selection",
        "test_data_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in payloads.items():
            bundle.writestr(name, content)
        bundle.writestr("input_manifest.json", json.dumps(input_manifest, indent=2))
    with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
        if "notebooks/train_lstm_colab.ipynb" not in source_zip.namelist():
            raise ValueError("Snapshot must include the thin Colab notebook.")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_commit": commit,
                "sha256": compute_file_sha256(args.output),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
