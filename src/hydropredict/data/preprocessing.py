"""Time-series preprocessing, sampling cadence assessment, and robust scaling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS, REQUIRED_COLUMNS, TIMESTAMP_COLUMN


@dataclass(frozen=True, slots=True)
class TelemetryGap:
    """Detected discontinuity in telemetry sampling."""

    start: pd.Timestamp
    end: pd.Timestamp
    duration_seconds: float
    missing_steps_estimate: int


@dataclass(frozen=True, slots=True)
class SamplingCadenceReport:
    """Summary of time-series continuity and sampling regularity."""

    total_records: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    median_step_seconds: float
    mean_step_seconds: float
    min_step_seconds: float
    max_step_seconds: float
    gap_count: int
    gaps: tuple[TelemetryGap, ...]

    def summary(self) -> dict[str, str | int | float]:
        return {
            "total_records": self.total_records,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "median_step_seconds": round(self.median_step_seconds, 2),
            "mean_step_seconds": round(self.mean_step_seconds, 2),
            "min_step_seconds": round(self.min_step_seconds, 2),
            "max_step_seconds": round(self.max_step_seconds, 2),
            "gap_count": self.gap_count,
        }


def load_telemetry_dataframe(source: Path | str) -> pd.DataFrame:
    """Load a validated telemetry CSV into a sorted pandas DataFrame.

    Ensures the timestamp column is parsed as datetime and set as index,
    and measurement columns are float64.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True)
    df = df.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)

    for col in MEASUREMENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(np.float64)

    return df


def assess_sampling_cadence(
    df: pd.DataFrame,
    nominal_step_seconds: float = 300.0,
    gap_threshold_factor: float = 1.5,
) -> SamplingCadenceReport:
    """Analyze time delta intervals between successive telemetry samples."""
    if len(df) < 2:
        raise ValueError("At least 2 observations are required to assess sampling cadence.")

    timestamps = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True).sort_values()
    deltas = timestamps.diff().dropna().dt.total_seconds()

    median_step = float(deltas.median())
    mean_step = float(deltas.mean())
    min_step = float(deltas.min())
    max_step = float(deltas.max())

    threshold = nominal_step_seconds * gap_threshold_factor
    gap_indices = deltas[deltas > threshold].index

    gaps: list[TelemetryGap] = []
    for idx in gap_indices:
        t_start = timestamps.loc[idx - 1]
        t_end = timestamps.loc[idx]
        duration = float(deltas.loc[idx])
        missing_steps = max(0, int(round(duration / nominal_step_seconds)) - 1)
        gaps.append(
            TelemetryGap(
                start=t_start,
                end=t_end,
                duration_seconds=duration,
                missing_steps_estimate=missing_steps,
            )
        )

    return SamplingCadenceReport(
        total_records=len(df),
        start_time=timestamps.iloc[0],
        end_time=timestamps.iloc[-1],
        median_step_seconds=median_step,
        mean_step_seconds=mean_step,
        min_step_seconds=min_step,
        max_step_seconds=max_step,
        gap_count=len(gaps),
        gaps=tuple(gaps),
    )


def resample_regular_grid(
    df: pd.DataFrame,
    freq: str = "5min",
    max_fill_steps: int = 3,
    method: Literal["linear", "forward"] = "linear",
) -> pd.DataFrame:
    """Resample telemetry onto an exact regular temporal grid.

    - Snaps timestamps to the specified frequency grid.
    - Interpolates small gaps up to `max_fill_steps` intervals.
    - Large gaps (> `max_fill_steps`) remain NaN to avoid synthetic hallucinations.
    - Adds an `is_interpolated` boolean column.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], utc=True)
    work = work.sort_values(by=TIMESTAMP_COLUMN).drop_duplicates(
        subset=[TIMESTAMP_COLUMN], keep="last"
    )

    dt_index = pd.DatetimeIndex(work[TIMESTAMP_COLUMN])
    work = work.set_index(dt_index)

    # Floor the start and ceil the end to ensure clean grid boundaries
    start = dt_index.min().floor(freq)
    end = dt_index.max().ceil(freq)
    grid_index = pd.date_range(
        start=start, end=end, freq=freq, tz=dt_index.tz, name=TIMESTAMP_COLUMN
    )

    # Reindex onto regular grid
    # To handle raw timestamps that have small second-level jitters, reindex nearest within 1.5 min
    reindexed = work.reindex(grid_index, method="nearest", tolerance=pd.Timedelta("90s"))

    # Track missing indicator before interpolation
    was_missing = reindexed[MEASUREMENT_COLUMNS[0]].isna()

    # Interpolate small gaps
    if method == "linear":
        interpolated = reindexed[list(MEASUREMENT_COLUMNS)].interpolate(
            method="time", limit=max_fill_steps, limit_direction="forward"
        )
    else:
        interpolated = reindexed[list(MEASUREMENT_COLUMNS)].ffill(limit=max_fill_steps)

    interpolated["is_interpolated"] = was_missing & interpolated[MEASUREMENT_COLUMNS[0]].notna()

    result = interpolated.reset_index()
    return result


class RobustTelemetryScaler:
    """Robust scaler using median and Interquartile Range (IQR).

    Preserves state for production inference or test set evaluation
    without causing data leakage.
    Formula: z = (x - median) / IQR
    """

    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.iqrs: dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(
        self, df: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
    ) -> RobustTelemetryScaler:
        """Compute median and IQR for target measurement columns."""
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame for scaling.")
            series = df[col].dropna()
            if series.empty:
                raise ValueError(f"Column '{col}' contains no valid data for scaling.")
            median_val = float(series.median())
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr_val = q75 - q25

            # If IQR is zero (e.g. constant signal), fallback to standard deviation or 1.0
            if iqr_val <= 1e-8:
                std_val = float(series.std())
                iqr_val = std_val if std_val > 1e-8 else 1.0

            self.medians[col] = median_val
            self.iqrs[col] = iqr_val

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformation to DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("RobustTelemetryScaler must be fitted before transforming data.")

        scaled = df.copy()
        for col, median_val in self.medians.items():
            if col in scaled.columns:
                iqr_val = self.iqrs[col]
                scaled[col] = (scaled[col] - median_val) / iqr_val

        return scaled

    def fit_transform(
        self, df: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
    ) -> pd.DataFrame:
        """Fit scaler and transform the DataFrame in one step."""
        return self.fit(df, columns).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revert scaled columns back to original engineering units."""
        if not self.is_fitted:
            raise RuntimeError("RobustTelemetryScaler must be fitted before inverse transform.")

        unscaled = df.copy()
        for col, median_val in self.medians.items():
            if col in unscaled.columns:
                iqr_val = self.iqrs[col]
                unscaled[col] = (unscaled[col] * iqr_val) + median_val

        return unscaled

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Serialize parameters for model artifact persistence."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted scaler.")
        return {
            "medians": self.medians,
            "iqrs": self.iqrs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> RobustTelemetryScaler:
        """Instantiate scaler from serialized parameters."""
        scaler = cls()
        scaler.medians = {k: float(v) for k, v in data.get("medians", {}).items()}
        scaler.iqrs = {k: float(v) for k, v in data.get("iqrs", {}).items()}
        scaler.is_fitted = bool(scaler.medians and scaler.iqrs)
        return scaler

    def save(self, path: Path | str) -> None:
        """Save scaler parameters as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> RobustTelemetryScaler:
        """Load scaler from JSON file."""
        target = Path(path)
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
