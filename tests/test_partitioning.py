from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from hydropredict.cli import main
from hydropredict.data.partitioning import (
    PartitionManifest,
    create_shp_partitions,
)


def create_sample_telemetry(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "V1", "V2", "V3", "V4", "V5", "V6"])
        writer.writerows(rows)
    return path


def test_create_shp_partitions_chronological(tmp_path: Path) -> None:
    train_src = tmp_path / "raw_train.csv"
    test_src = tmp_path / "raw_test.csv"
    faults_src = tmp_path / "raw_faults.csv"
    out_dir = tmp_path / "partitions"

    # Train data: 2018-08 to 2019-01
    create_sample_telemetry(
        train_src,
        [
            ["2018-08-15T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["2019-01-15T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["2019-07-15T00:00:00Z", "1", "2", "3", "4", "5", "6"],  # will be filtered out of train
        ],
    )

    # Test data: 2019-03 (val) and 2019-06 (test)
    create_sample_telemetry(
        test_src,
        [
            ["2019-03-15T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["2019-06-15T00:00:00Z", "1", "2", "3", "4", "5", "6"],
        ],
    )

    # Faults: one in val, one in test
    with faults_src.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"])
        writer.writerow(["2019-03-16 00:00:00"])
        writer.writerow(["2019-06-16 00:00:00"])

    manifest = create_shp_partitions(
        train_source=train_src,
        test_source=test_src,
        faults_source=faults_src,
        output_dir=out_dir,
        strategy="chronological",
        train_end_boundary="2019-02-01T00:00:00Z",
        val_end_boundary="2019-05-01T00:00:00Z",
    )

    assert manifest.strategy == "chronological"
    assert manifest.is_strictly_chronological is True
    assert set(manifest.partitions.keys()) == {"train", "validation", "test"}

    # Train partition has the 2 records before 2019-02-01
    assert manifest.partitions["train"].record_count == 2
    # Val partition has the 1 record in March 2019
    assert manifest.partitions["validation"].record_count == 1
    assert manifest.partitions["validation"].fault_events_count == 1
    # Test partition has the 1 record in June 2019
    assert manifest.partitions["test"].record_count == 1
    assert manifest.partitions["test"].fault_events_count == 1

    # Verify strict non-leakage
    train_max = pd.Timestamp(manifest.partitions["train"].end_time)
    val_min = pd.Timestamp(manifest.partitions["validation"].start_time)
    test_min = pd.Timestamp(manifest.partitions["test"].start_time)
    assert train_max < val_min < test_min

    # Verify manifest saved and reloadable
    manifest_file = out_dir / "partitions_manifest.json"
    assert manifest_file.exists()
    reloaded = PartitionManifest.load(manifest_file)
    assert reloaded.strategy == manifest.strategy
    assert reloaded.partitions["train"].sha256 == manifest.partitions["train"].sha256


def test_create_shp_partitions_published(tmp_path: Path) -> None:
    train_src = tmp_path / "raw_train.csv"
    test_src = tmp_path / "raw_test.csv"
    faults_src = tmp_path / "raw_faults.csv"
    out_dir = tmp_path / "partitions_pub"

    create_sample_telemetry(train_src, [["2018-08-15T00:00:00Z", "1", "2", "3", "4", "5", "6"]])
    create_sample_telemetry(test_src, [["2018-08-16T00:00:00Z", "1", "2", "3", "4", "5", "6"]])

    with faults_src.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"])
        writer.writerow(["2018-08-17 00:00:00"])

    manifest = create_shp_partitions(
        train_source=train_src,
        test_source=test_src,
        faults_source=faults_src,
        output_dir=out_dir,
        strategy="published",
    )

    assert manifest.strategy == "published"
    assert manifest.is_strictly_chronological is False
    assert set(manifest.partitions.keys()) == {"train", "test"}
    assert manifest.partitions["train"].record_count == 1
    assert manifest.partitions["test"].record_count == 1


def test_cli_partition(tmp_path: Path, capsys: None) -> None:
    train_src = tmp_path / "raw_train.csv"
    test_src = tmp_path / "raw_test.csv"
    faults_src = tmp_path / "raw_faults.csv"
    out_dir = tmp_path / "cli_partitions"

    create_sample_telemetry(train_src, [["2018-08-15T00:00:00Z", "1", "2", "3", "4", "5", "6"]])
    create_sample_telemetry(test_src, [["2019-06-15T00:00:00Z", "1", "2", "3", "4", "5", "6"]])
    with faults_src.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"])

    exit_code = main(
        [
            "partition",
            "--train-source",
            str(train_src),
            "--test-source",
            str(test_src),
            "--faults-source",
            str(faults_src),
            "--output-dir",
            str(out_dir),
            "--strategy",
            "chronological",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "partitions_manifest.json").exists()
