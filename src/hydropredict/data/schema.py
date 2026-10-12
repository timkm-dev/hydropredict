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


# Sensor semantics verified against de Santis & Costa (2020) and project report specifications
SENSOR_METADATA: dict[str, SensorChannelMetadata] = {
    "V1": SensorChannelMetadata(
        channel_id="V1",
        description="Generator guide bearing horizontal radial vibration",
        physical_quantity="vibration_rms",
        unit="mm/s",
        typical_min=0.0,
        typical_max=10.0,
    ),
    "V2": SensorChannelMetadata(
        channel_id="V2",
        description="Generator guide bearing vertical radial vibration",
        physical_quantity="vibration_rms",
        unit="mm/s",
        typical_min=0.0,
        typical_max=10.0,
    ),
    "V3": SensorChannelMetadata(
        channel_id="V3",
        description="Turbine guide bearing radial vibration",
        physical_quantity="vibration_rms",
        unit="mm/s",
        typical_min=0.0,
        typical_max=10.0,
    ),
    "V4": SensorChannelMetadata(
        channel_id="V4",
        description="Thrust bearing axial vibration",
        physical_quantity="vibration_rms",
        unit="mm/s",
        typical_min=0.0,
        typical_max=10.0,
    ),
    "V5": SensorChannelMetadata(
        channel_id="V5",
        description="Bearing lubrication operating temperature",
        physical_quantity="temperature",
        unit="°C",
        typical_min=0.0,
        typical_max=100.0,
    ),
    "V6": SensorChannelMetadata(
        channel_id="V6",
        description="Generator active electrical power output",
        physical_quantity="active_power",
        unit="kW",
        typical_min=0.0,
        typical_max=10000.0,
    ),
}


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
