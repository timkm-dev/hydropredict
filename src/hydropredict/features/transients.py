"""Transient detection and numerical rate-of-change feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS, TIMESTAMP_COLUMN


def compute_transient_features(
    df: pd.DataFrame,
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
    time_column: str = TIMESTAMP_COLUMN,
    nominal_step_seconds: float = 300.0,
) -> pd.DataFrame:
    """Compute first and second discrete numerical time derivatives.

    These capture rapid mechanical load fluctuations and electrical shocks.
    If 'segment_id' is present, transient derivatives reset at segment boundaries.
    Metrics computed per measurement:
      - rate_of_change: dV/dt (units per second)
      - abs_rate_of_change: |dV/dt|
      - acceleration: d^2V/dt^2 (rate of change of derivative)
    """
    features = pd.DataFrame(index=df.index)
    has_segments = "segment_id" in df.columns

    if not has_segments:
        if time_column in df.columns:
            dt_series = pd.to_datetime(df[time_column], utc=True).diff().dt.total_seconds()
            dt_series = dt_series.fillna(nominal_step_seconds)
            dt = np.where(dt_series > 0, dt_series, nominal_step_seconds)
        else:
            dt = np.full(len(df), nominal_step_seconds)

        for col in columns:
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=np.float64)
            diff1 = np.diff(vals, prepend=vals[0])
            roc = diff1 / dt
            diff2 = np.diff(roc, prepend=roc[0])
            acc = diff2 / dt
            features[f"{col}_roc"] = roc
            features[f"{col}_abs_roc"] = np.abs(roc)
            features[f"{col}_acc"] = acc
        return features

    # Segment-aware calculation
    roc_dict: dict[str, list[float]] = {f"{col}_roc": [] for col in columns if col in df.columns}
    abs_roc_dict: dict[str, list[float]] = {
        f"{col}_abs_roc": [] for col in columns if col in df.columns
    }
    acc_dict: dict[str, list[float]] = {f"{col}_acc": [] for col in columns if col in df.columns}

    for _, group in df.groupby("segment_id", sort=False):
        if time_column in group.columns:
            g_dt = pd.to_datetime(group[time_column], utc=True).diff().dt.total_seconds()
            g_dt = g_dt.fillna(nominal_step_seconds)
            dt_arr = np.where(g_dt > 0, g_dt, nominal_step_seconds)
        else:
            dt_arr = np.full(len(group), nominal_step_seconds)

        for col in columns:
            if col not in group.columns:
                continue
            vals = group[col].to_numpy(dtype=np.float64)
            diff1 = np.diff(vals, prepend=vals[0])
            roc = diff1 / dt_arr
            diff2 = np.diff(roc, prepend=roc[0])
            acc = diff2 / dt_arr
            roc_dict[f"{col}_roc"].extend(roc)
            abs_roc_dict[f"{col}_abs_roc"].extend(np.abs(roc))
            acc_dict[f"{col}_acc"].extend(acc)

    for col in columns:
        if col not in df.columns:
            continue
        features[f"{col}_roc"] = roc_dict[f"{col}_roc"]
        features[f"{col}_abs_roc"] = abs_roc_dict[f"{col}_abs_roc"]
        features[f"{col}_acc"] = acc_dict[f"{col}_acc"]

    return features
