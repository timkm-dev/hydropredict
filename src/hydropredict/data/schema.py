"""Canonical telemetry schema and sensor metadata used throughout HydroPredict."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

TIMESTAMP_COLUMN = "t"
MEASUREMENT_COLUMNS = tuple(f"V{index}" for index in range(1, 7))
REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, *MEASUREMENT_COLUMNS)


@dataclass(frozen=True, slots=True)
class SensorChannelMetadata:
    """Authoritative physical description and engineering units for one telemetry channel."""

    channel_id: str
    description: str
    physical_quantity: str
    unit: str
    typical_min: float
    typical_max: float


# Channel order, units and physical limits require a source data dictionary.
# The paper names vibration, HLU inflow and apparent power, not temperature.
# Leave unverified physical metadata unset; numerical validation remains enforced.
SENSOR_METADATA: dict[str, SensorChannelMetadata] = {}


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

    def as_dict(self, include_source_row: bool = False) -> dict[str, Any]:
        values: dict[str, Any] = {TIMESTAMP_COLUMN: self.timestamp}
        values.update(dict(zip(MEASUREMENT_COLUMNS, self.measurements, strict=True)))
        if include_source_row:
            values["source_row"] = self.source_row
        return values
