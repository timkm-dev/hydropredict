from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hydropredict.data.validation import SchemaError, validate_csv, write_accepted


def write_csv(path: Path, rows: list[list[str]], header: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["t", "V1", "V2", "V3", "V4", "V5", "V6"])
        writer.writerows(rows)


def test_accepts_valid_telemetry(tmp_path: Path) -> None:
    source = tmp_path / "valid.csv"
    write_csv(
        source,
        [
            ["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["2026-01-01T00:05:00Z", "1.1", "2.1", "3.1", "4.1", "5.1", "6.1"],
        ],
    )

    report = validate_csv(source)

    assert report.total_rows == 2
    assert report.accepted_count == 2
    assert report.rejected_count == 0


def test_quarantines_invalid_and_duplicate_rows(tmp_path: Path) -> None:
    source = tmp_path / "mixed.csv"
    write_csv(
        source,
        [
            ["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["bad-date", "1", "2", "3", "4", "5", "6"],
            ["2026-01-01T00:05:00Z", "1", "2", "NaN", "4", "5", "6"],
            ["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"],
        ],
    )

    report = validate_csv(source)

    assert report.accepted_count == 1
    assert report.rejected_count == 3
    assert {issue.code for issue in report.issues} == {
        "duplicate_timestamp",
        "invalid_measurement",
        "invalid_timestamp",
    }


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    source = tmp_path / "missing_column.csv"
    write_csv(source, [], header=["t", "V1", "V2", "V3", "V4", "V5"])

    with pytest.raises(SchemaError, match="V6"):
        validate_csv(source)


def test_writes_canonical_accepted_file(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    write_csv(source, [["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"]])

    write_accepted(validate_csv(source), output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["t"] == "2026-01-01T00:00:00+00:00"
    assert rows[0]["V6"] == "6.0"
