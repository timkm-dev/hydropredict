"""CSV validation for hydroelectric telemetry."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from hydropredict.data.schema import (
    MEASUREMENT_COLUMNS,
    REQUIRED_COLUMNS,
    SENSOR_METADATA,
    TIMESTAMP_COLUMN,
    TelemetryRecord,
)


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

    def to_dataframe(self, include_source_row: bool = False) -> pd.DataFrame:
        """Convert accepted records into a sorted pandas DataFrame."""
        if not self.accepted:
            cols = list(REQUIRED_COLUMNS)
            if include_source_row:
                cols.append("source_row")
            empty_df = pd.DataFrame(columns=cols)
            empty_df[TIMESTAMP_COLUMN] = pd.to_datetime(empty_df[TIMESTAMP_COLUMN], utc=True)
            for c in MEASUREMENT_COLUMNS:
                empty_df[c] = empty_df[c].astype(np.float64)
            return empty_df

        data = [record.as_dict(include_source_row=include_source_row) for record in self.accepted]
        df = pd.DataFrame.from_records(data)
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True)
        df = df.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
        for col in MEASUREMENT_COLUMNS:
            df[col] = df[col].astype(np.float64)
        return df


def _parse_timestamp(value: str) -> datetime:
    normalised = value.strip()
    if not normalised:
        raise ValueError("timestamp is empty")
    if normalised.endswith("Z"):
        normalised = f"{normalised[:-1]}+00:00"
    ts = datetime.fromisoformat(normalised)
    # Standardize timezone-naive plant telemetry timestamps to UTC per HydroPredict contract
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


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


def check_physical_plausibility(
    record: TelemetryRecord,
) -> list[tuple[str, str]]:
    """Flag physical plausibility warnings without rejecting unusual operational regimes."""
    warnings: list[tuple[str, str]] = []
    for col, val in zip(MEASUREMENT_COLUMNS, record.measurements, strict=True):
        meta = SENSOR_METADATA.get(col)
        if meta is None:
            continue
        if val < meta.typical_min or val > meta.typical_max:
            msg = (
                f"{col} value {val} is outside typical range "
                f"[{meta.typical_min}, {meta.typical_max}] {meta.unit}"
            )
            warnings.append(("range_warning", msg))
    return warnings


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
