"""End-to-end feature extraction pipeline for HydroPredict telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS, TIMESTAMP_COLUMN
from hydropredict.features.domain import compute_domain_features
from hydropredict.features.temporal import compute_rolling_features
from hydropredict.features.transients import compute_transient_features

# Evaluation and metadata columns that must NEVER enter model predictor inputs
EXCLUDED_EVALUATION_COLUMNS: frozenset[str] = frozenset(
    {
        "fault_state",
        "time_to_next_fault_seconds",
        "nearest_fault_distance_seconds",
        "source_row",
    }
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Configuration options for feature extraction."""

    include_raw: bool = True
    rolling_windows: tuple[int, ...] = (6, 24)
    include_transients: bool = True
    include_domain: bool = True
    min_power_floor: float = 50.0
    include_segment_id: bool = True


def extract_feature_matrix(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Transform telemetry DataFrame into a comprehensive engineered feature matrix.

    Temporal ordering is preserved and all feature calculations are strictly causal
    (backward-looking) to prevent data leakage. Ground-truth fault labels and
    evaluation metadata are strictly excluded from predictor outputs.
    """
    cfg = config or FeatureConfig()
    if TIMESTAMP_COLUMN in df.columns:
        times = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True)
        if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError("Feature timestamps must be valid, unique and chronological.")
    feature_parts: list[pd.DataFrame] = []

    if cfg.include_raw:
        raw_cols = [c for c in MEASUREMENT_COLUMNS if c in df.columns]
        feature_parts.append(df[raw_cols].copy())

    if cfg.rolling_windows:
        rolling_df = compute_rolling_features(df, windows=cfg.rolling_windows)
        feature_parts.append(rolling_df)

    if cfg.include_transients:
        transient_df = compute_transient_features(df)
        feature_parts.append(transient_df)

    if cfg.include_domain:
        domain_df = compute_domain_features(df, min_power_floor=cfg.min_power_floor)
        feature_parts.append(domain_df)

    matrix = pd.concat(feature_parts, axis=1)

    # Ensure no evaluation labels or leakage columns entered feature columns
    leakage_cols = set(matrix.columns).intersection(EXCLUDED_EVALUATION_COLUMNS)
    if leakage_cols:
        raise RuntimeError(
            f"Data leakage detected! Evaluation columns entered predictor matrix: {leakage_cols}"
        )

    # Attach structural coordinate columns: timestamp and segment_id
    if cfg.include_segment_id and "segment_id" in df.columns:
        matrix.insert(0, "segment_id", df["segment_id"].to_numpy())

    if TIMESTAMP_COLUMN in df.columns:
        matrix.insert(0, TIMESTAMP_COLUMN, df[TIMESTAMP_COLUMN].to_numpy())

    return matrix
