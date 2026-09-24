"""Canonical telemetry schema used throughout HydroPredict."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

TIMESTAMP_COLUMN = "t"
MEASUREMENT_COLUMNS = tuple(f"V{index}" for index in range(1, 7))
REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, *MEASUREMENT_COLUMNS)


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """One validated source observation."""

    timestamp: datetime
    measurements: tuple[float, float, float, float, float, float]
    source_row: int

    def as_csv_row(self) -> dict[str, str | float]:
        values: dict[str, str | float] = {TIMESTAMP_COLUMN: self.timestamp.isoformat()}
        values.update(dict(zip(MEASUREMENT_COLUMNS, self.measurements, strict=True)))
        return values
