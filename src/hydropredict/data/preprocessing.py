"""Time-series preprocessing, sampling cadence assessment, and robust scaling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS, TIMESTAMP_COLUMN
from hydropredict.data.validation import validate_csv


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


@dataclass(frozen=True, slots=True)
class SegmentMetadata:
    """Metadata describing one continuous, uninterrupted telemetry segment."""

    segment_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    record_count: int
    imputed_count: int


@dataclass(frozen=True, slots=True)
class PreparationAccountingReport:
    """Auditable row and step accounting across preprocessing transformations."""

    total_raw_records: int
    retained_records: int
    collision_count: int
    segment_count: int
    imputed_step_count: int
    total_prepared_records: int
    outage_count: int
    total_outage_seconds: float
    segments: tuple[SegmentMetadata, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "total_raw_records": self.total_raw_records,
            "retained_records": self.retained_records,
            "collision_count": self.collision_count,
            "segment_count": self.segment_count,
            "imputed_step_count": self.imputed_step_count,
            "total_prepared_records": self.total_prepared_records,
            "outage_count": self.outage_count,
            "total_outage_seconds": round(self.total_outage_seconds, 2),
        }


def load_telemetry_dataframe(
    source: Path | str,
    strict: bool = True,
    include_source_row: bool = False,
) -> pd.DataFrame:
    """Load a validated telemetry CSV into a sorted pandas DataFrame.

    Routes through `validate_csv` to enforce the single canonical telemetry
    contract (required schema, valid timestamps, numeric finite values, and
    duplicate detection).
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    report = validate_csv(path)
    if strict and report.rejected_count > 0:
        first_issue = report.issues[0]
        msg = (
            f"Validation failed for {path}: {report.rejected_count} row(s) rejected. "
            f"First issue at row {first_issue.row_number}: "
            f"[{first_issue.code}] {first_issue.message}"
        )
        raise ValueError(msg)

    return report.to_dataframe(include_source_row=include_source_row)


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


def prepare_telemetry_series(
    df: pd.DataFrame,
    freq: str = "5min",
    max_fill_steps: int = 3,
    method: Literal["linear", "forward"] = "forward",
    nominal_step_seconds: float | None = None,
    binning: Literal["ceil", "round"] = "ceil",
    keep_collision: Literal["last", "first"] = "last",
) -> tuple[pd.DataFrame, PreparationAccountingReport]:
    """Prepare telemetry into regular, continuous segments with causal gap handling.

    - Bins observations into uniform time epochs without nearest-neighbor lookahead.
    - Resolves collisions while preserving original timestamps and accounting for overwritten rows.
    - Identifies continuous segments bounded by equipment outages (> max_fill_steps * nominal_step).
    - Fills short gaps within segments causally using bounded forward fill (or linear).
    - Excludes extended outages from the grid to prevent grid inflation and synthetic hallucination.
    - Returns prepared DataFrame and a full PreparationAccountingReport.
    """
    step_seconds = pd.Timedelta(freq).total_seconds()
    if step_seconds <= 0 or max_fill_steps < 0:
        raise ValueError("Frequency must be positive and max_fill_steps must be nonnegative.")
    if nominal_step_seconds is not None and nominal_step_seconds != step_seconds:
        raise ValueError("nominal_step_seconds must match freq.")
    if df.empty:
        empty_cols = [
            TIMESTAMP_COLUMN,
            *MEASUREMENT_COLUMNS,
            "segment_id",
            "is_interpolated",
            "t_original",
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_report = PreparationAccountingReport(
            total_raw_records=0,
            retained_records=0,
            collision_count=0,
            segment_count=0,
            imputed_step_count=0,
            total_prepared_records=0,
            outage_count=0,
            total_outage_seconds=0.0,
            segments=(),
        )
        return empty_df, empty_report

    work = df.copy()
    work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], utc=True)
    work["t_original"] = work[TIMESTAMP_COLUMN]

    # Map timestamps to regular frequency bin
    if binning == "ceil":
        work["t_bin"] = work[TIMESTAMP_COLUMN].dt.ceil(freq)
    else:
        work["t_bin"] = work[TIMESTAMP_COLUMN].dt.round(freq)

    work = work.sort_values(by=["t_bin", "t_original"]).reset_index(drop=True)
    deduped = work.drop_duplicates(subset=["t_bin"], keep=keep_collision).copy()
    collision_count = len(work) - len(deduped)

    # Detect segment boundaries where gap exceeds max_fill_steps intervals
    deltas = deduped["t_bin"].diff().dt.total_seconds()
    gap_threshold = (max_fill_steps + 1) * step_seconds
    is_new_segment = (deltas > gap_threshold) | deltas.isna()
    deduped["segment_id"] = (is_new_segment.cumsum() - 1).astype(int)

    outage_deltas = deltas[deltas > gap_threshold]
    total_outage_seconds = float(outage_deltas.sum()) if not outage_deltas.empty else 0.0
    outage_count = len(outage_deltas)

    prepared_segments: list[pd.DataFrame] = []
    segment_meta_list: list[SegmentMetadata] = []
    total_imputed = 0

    for seg_id, group in deduped.groupby("segment_id", sort=True):
        seg_start = group["t_bin"].iloc[0]
        seg_end = group["t_bin"].iloc[-1]
        grid = pd.date_range(start=seg_start, end=seg_end, freq=freq, tz=UTC, name=TIMESTAMP_COLUMN)

        indexed = group.set_index("t_bin")
        reindexed = indexed.reindex(grid)

        was_missing = reindexed[MEASUREMENT_COLUMNS[0]].isna()
        if method == "forward":
            filled_measurements = reindexed[list(MEASUREMENT_COLUMNS)].ffill()
        else:
            filled_measurements = reindexed[list(MEASUREMENT_COLUMNS)].interpolate(method="time")

        seg_id_int = int(str(seg_id))
        filled_measurements["is_interpolated"] = was_missing
        filled_measurements["segment_id"] = seg_id_int
        filled_measurements["t_original"] = reindexed["t_original"]

        # For imputed steps, assign effective grid timestamp to t_original
        filled_measurements.loc[was_missing, "t_original"] = filled_measurements.index[was_missing]

        seg_imputed_count = int(was_missing.sum())
        total_imputed += seg_imputed_count

        seg_df = filled_measurements.reset_index().rename(columns={"index": TIMESTAMP_COLUMN})
        prepared_segments.append(seg_df)

        segment_meta_list.append(
            SegmentMetadata(
                segment_id=seg_id_int,
                start_time=seg_start,
                end_time=seg_end,
                record_count=len(seg_df),
                imputed_count=seg_imputed_count,
            )
        )

    prepared = (
        pd.concat(prepared_segments, ignore_index=True) if prepared_segments else pd.DataFrame()
    )

    report = PreparationAccountingReport(
        total_raw_records=len(df),
        retained_records=len(deduped),
        collision_count=collision_count,
        segment_count=len(segment_meta_list),
        imputed_step_count=total_imputed,
        total_prepared_records=len(prepared),
        outage_count=outage_count,
        total_outage_seconds=total_outage_seconds,
        segments=tuple(segment_meta_list),
    )

    return prepared, report


def resample_regular_grid(
    df: pd.DataFrame,
    freq: str = "5min",
    max_fill_steps: int = 3,
    method: Literal["linear", "forward"] = "forward",
) -> pd.DataFrame:
    """Resample telemetry onto an exact regular temporal grid with segment preservation."""
    prepared, _ = prepare_telemetry_series(
        df=df,
        freq=freq,
        max_fill_steps=max_fill_steps,
        method=method,
    )
    return prepared


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters for model artifact persistence."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted scaler.")
        return {
            "scaler_type": "robust",
            "medians": self.medians,
            "iqrs": self.iqrs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustTelemetryScaler:
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


class StandardTelemetryScaler:
    """Standard z-score scaler using mean and standard deviation.

    Formula: z = (x - mean) / std
    """

    def __init__(self) -> None:
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(
        self, df: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
    ) -> StandardTelemetryScaler:
        """Compute mean and standard deviation for target measurement columns."""
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame for scaling.")
            series = df[col].dropna()
            if series.empty:
                raise ValueError(f"Column '{col}' contains no valid data for scaling.")
            mean_val = float(series.mean())
            std_val = float(series.std())
            if std_val <= 1e-8:
                std_val = 1.0

            self.means[col] = mean_val
            self.stds[col] = std_val

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard z-score scaling transformation."""
        if not self.is_fitted:
            raise RuntimeError("StandardTelemetryScaler must be fitted before transforming data.")

        scaled = df.copy()
        for col, mean_val in self.means.items():
            if col in scaled.columns:
                std_val = self.stds[col]
                scaled[col] = (scaled[col] - mean_val) / std_val

        return scaled

    def fit_transform(
        self, df: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_COLUMNS
    ) -> pd.DataFrame:
        """Fit scaler and transform the DataFrame in one step."""
        return self.fit(df, columns).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revert scaled columns back to original engineering units."""
        if not self.is_fitted:
            raise RuntimeError("StandardTelemetryScaler must be fitted before inverse transform.")

        unscaled = df.copy()
        for col, mean_val in self.means.items():
            if col in unscaled.columns:
                std_val = self.stds[col]
                unscaled[col] = (unscaled[col] * std_val) + mean_val

        return unscaled

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters for model artifact persistence."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted scaler.")
        return {
            "scaler_type": "standard",
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardTelemetryScaler:
        """Instantiate scaler from serialized parameters."""
        scaler = cls()
        scaler.means = {k: float(v) for k, v in data.get("means", {}).items()}
        scaler.stds = {k: float(v) for k, v in data.get("stds", {}).items()}
        scaler.is_fitted = bool(scaler.means and scaler.stds)
        return scaler

    def save(self, path: Path | str) -> None:
        """Save scaler parameters as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> StandardTelemetryScaler:
        """Load scaler from JSON file."""
        target = Path(path)
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


def load_scaler(path: Path | str) -> RobustTelemetryScaler | StandardTelemetryScaler:
    """Load scaler artifact auto-detecting robust vs standard type."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    scaler_type = data.get("scaler_type", "robust")
    if scaler_type == "standard":
        return StandardTelemetryScaler.from_dict(data)
    return RobustTelemetryScaler.from_dict(data)
