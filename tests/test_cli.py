from __future__ import annotations

import csv
from pathlib import Path

from hydropredict.cli import main


def write_test_csv(path: Path, rows: list[list[str]], header: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_cli_validate(tmp_path: Path, capsys: object) -> None:
    csv_file = tmp_path / "valid.csv"
    write_test_csv(
        csv_file,
        [["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"]],
        header=["t", "V1", "V2", "V3", "V4", "V5", "V6"],
    )

    exit_code = main(["validate", str(csv_file)])
    assert exit_code == 0


def test_cli_preprocess_and_features(tmp_path: Path) -> None:
    input_csv = tmp_path / "raw_telemetry.csv"
    preprocessed_csv = tmp_path / "preprocessed.csv"
    features_csv = tmp_path / "features.csv"
    scaler_json = tmp_path / "scaler.json"

    write_test_csv(
        input_csv,
        [
            ["2026-01-01T00:00:00Z", "0.2", "0.1", "0.3", "0.1", "18.0", "3000.0"],
            ["2026-01-01T00:05:00Z", "0.3", "0.15", "0.4", "0.12", "18.2", "3100.0"],
        ],
        header=["t", "V1", "V2", "V3", "V4", "V5", "V6"],
    )

    # 1. preprocess
    prep_code = main(
        [
            "preprocess",
            str(input_csv),
            "--output",
            str(preprocessed_csv),
            "--scaler-output",
            str(scaler_json),
        ]
    )
    assert prep_code == 0
    assert preprocessed_csv.exists()
    assert scaler_json.exists()

    # 2. extract-features
    feat_code = main(
        [
            "extract-features",
            str(preprocessed_csv),
            "--output",
            str(features_csv),
            "--windows",
            "2",
        ]
    )
    assert feat_code == 0
    assert features_csv.exists()


def test_cli_preprocess_fit_and_reuse_scaler(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_prep = tmp_path / "train_prep.csv"
    val_prep = tmp_path / "val_prep.csv"
    scaler_file = tmp_path / "fitted_scaler.json"
    accounting_file = tmp_path / "accounting.json"

    write_test_csv(
        train_csv,
        [
            ["2026-01-01T00:00:00Z", "10.0", "1.0", "1.0", "1.0", "20.0", "3000.0"],
            ["2026-01-01T00:05:00Z", "20.0", "1.0", "1.0", "1.0", "20.0", "3000.0"],
        ],
        header=["t", "V1", "V2", "V3", "V4", "V5", "V6"],
    )
    write_test_csv(
        val_csv,
        [
            ["2026-01-01T01:00:00Z", "100.0", "1.0", "1.0", "1.0", "20.0", "3000.0"],
        ],
        header=["t", "V1", "V2", "V3", "V4", "V5", "V6"],
    )

    # 1. Preprocess train: fit standard scaler, apply scaling, export accounting report
    train_code = main(
        [
            "preprocess",
            str(train_csv),
            "--output",
            str(train_prep),
            "--fit-scaler",
            str(scaler_file),
            "--scaler-type",
            "standard",
            "--scale",
            "--accounting-output",
            str(accounting_file),
        ]
    )
    assert train_code == 0
    assert train_prep.exists()
    assert scaler_file.exists()
    assert accounting_file.exists()

    # 2. Preprocess val: load training scaler, apply scaling without refitting
    val_code = main(
        [
            "preprocess",
            str(val_csv),
            "--output",
            str(val_prep),
            "--scaler",
            str(scaler_file),
            "--scale",
        ]
    )
    assert val_code == 0
    assert val_prep.exists()

    # 3. Requesting scaling without scaler fails with exit code 2
    err_code = main(
        [
            "preprocess",
            str(val_csv),
            "--scale",
        ]
    )
    assert err_code == 2


def test_cli_align_faults_and_eda(tmp_path: Path) -> None:
    tel_csv = tmp_path / "tel.csv"
    faults_csv = tmp_path / "faults.csv"
    labeled_csv = tmp_path / "labeled.csv"

    write_test_csv(
        tel_csv,
        [
            ["2026-01-01T00:00:00Z", "0.2", "0.1", "0.3", "0.1", "18.0", "3000.0"],
            ["2026-01-01T01:00:00Z", "0.5", "0.3", "0.7", "0.2", "19.5", "3000.0"],
        ],
        header=["t", "V1", "V2", "V3", "V4", "V5", "V6"],
    )
    write_test_csv(faults_csv, [["2026-01-01T01:00:00Z"]], header=["t"])

    # 1. align-faults
    align_code = main(
        [
            "align-faults",
            str(tel_csv),
            str(faults_csv),
            "--output",
            str(labeled_csv),
            "--precursor-hours",
            "2.0",
        ]
    )
    assert align_code == 0
    assert labeled_csv.exists()

    # 2. eda
    eda_code = main(["eda", str(labeled_csv), "--json"])
    assert eda_code == 0

    eda_text_code = main(["eda", str(labeled_csv)])
    assert eda_text_code == 0


def test_cli_error_handling(tmp_path: Path) -> None:
    # Non-existent file error
    exit_code = main(["preprocess", str(tmp_path / "nonexistent.csv")])
    assert exit_code == 2
