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


def test_standardizes_timezone_naive_timestamps_to_utc(tmp_path: Path) -> None:
    source = tmp_path / "naive.csv"
    write_csv(
        source, [["2018-08-13 13:44:40.130", "0.16", "0.1", "0.36", "0.14", "13.69", "2736.0"]]
    )

    report = validate_csv(source)
    assert report.accepted_count == 1
    record = report.accepted[0]
    assert record.timestamp.tzinfo is not None
    assert str(record.timestamp.tzinfo) == "UTC"


def test_validation_report_to_dataframe(tmp_path: Path) -> None:
    source = tmp_path / "telemetry.csv"
    write_csv(
        source,
        [
            ["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["2026-01-01T00:05:00Z", "1.1", "2.1", "3.1", "4.1", "5.1", "6.1"],
        ],
    )

    report = validate_csv(source)
    df = report.to_dataframe(include_source_row=True)

    assert len(df) == 2
    assert "source_row" in df.columns
    assert list(df["source_row"]) == [2, 3]
    assert df["V1"].iloc[0] == 1.0


def test_physical_plausibility_warnings(tmp_path: Path) -> None:
    from hydropredict.data.validation import check_physical_plausibility

    source = tmp_path / "extreme.csv"
    # V5 = 120 °C (typical max 100), V6 = 15000 kW (typical max 10000)
    write_csv(source, [["2026-01-01T00:00:00Z", "0.5", "0.5", "0.5", "0.5", "120.0", "15000.0"]])

    report = validate_csv(source)
    assert report.accepted_count == 1
    warnings = check_physical_plausibility(report.accepted[0])
    assert len(warnings) == 2
    assert any("V5" in w[1] for w in warnings)
    assert any("V6" in w[1] for w in warnings)
