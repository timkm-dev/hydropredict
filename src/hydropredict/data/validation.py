"""CSV validation for hydroelectric telemetry."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from hydropredict.data.schema import MEASUREMENT_COLUMNS, REQUIRED_COLUMNS, TelemetryRecord


class SchemaError(ValueError):
    """Raised when a file cannot satisfy the telemetry input contract."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Reason a source row was rejected."""

    row_number: int
    code: str
    message: str
    source_values: dict[str, str]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Accepted records and row-level validation issues."""

    source: Path
    total_rows: int
    accepted: tuple[TelemetryRecord, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len({issue.row_number for issue in self.issues})

    def summary(self) -> dict[str, str | int]:
        return {
            "source": str(self.source),
            "total_rows": self.total_rows,
            "accepted_rows": self.accepted_count,
            "rejected_rows": self.rejected_count,
            "issue_count": len(self.issues),
        }


def _parse_timestamp(value: str) -> datetime:
    normalised = value.strip()
    if normalised.endswith("Z"):
        normalised = f"{normalised[:-1]}+00:00"
    if not normalised:
        raise ValueError("timestamp is empty")
    return datetime.fromisoformat(normalised)


def _parse_measurements(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
    parsed: list[float] = []
    for column in MEASUREMENT_COLUMNS:
        raw_value = (row.get(column) or "").strip()
        if not raw_value:
            raise ValueError(f"{column} is empty")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{column} must be finite")
        parsed.append(value)
    return tuple(parsed)  # type: ignore[return-value]


def _read_rows(handle: TextIO, source: Path) -> ValidationReport:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise SchemaError("CSV file has no header row")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SchemaError(f"CSV file is missing required columns: {missing}")

    accepted: list[TelemetryRecord] = []
    issues: list[ValidationIssue] = []
    seen_timestamps: set[datetime] = set()
    total_rows = 0

    for row_number, raw_row in enumerate(reader, start=2):
        total_rows += 1
        row = {key: value or "" for key, value in raw_row.items() if key is not None}
        row_issues: list[tuple[str, str]] = []

        try:
            timestamp = _parse_timestamp(row.get("t", ""))
        except ValueError as error:
            timestamp = None
            row_issues.append(("invalid_timestamp", str(error)))

        try:
            measurements = _parse_measurements(row)
        except ValueError as error:
            measurements = None
            row_issues.append(("invalid_measurement", str(error)))

        if timestamp is not None and timestamp in seen_timestamps:
            row_issues.append(("duplicate_timestamp", "timestamp duplicates an accepted row"))

        if row_issues:
            issues.extend(
                ValidationIssue(row_number, code, message, row) for code, message in row_issues
            )
            continue

        assert timestamp is not None
        assert measurements is not None
        seen_timestamps.add(timestamp)
        accepted.append(TelemetryRecord(timestamp, measurements, row_number))

    return ValidationReport(source, total_rows, tuple(accepted), tuple(issues))


def validate_csv(path: str | Path) -> ValidationReport:
    """Validate a UTF-8 CSV file against the canonical telemetry contract."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return _read_rows(handle, source)


def write_accepted(report: ValidationReport, destination: str | Path) -> None:
    """Write validated observations in canonical column order."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(record.as_csv_row() for record in report.accepted)


def write_quarantine(report: ValidationReport, destination: str | Path) -> None:
    """Write validation issues with the original source values."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("row_number", "code", "message", *REQUIRED_COLUMNS)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for issue in report.issues:
            writer.writerow(
                {
                    "row_number": issue.row_number,
                    "code": issue.code,
                    "message": issue.message,
                    **issue.source_values,
                }
            )
