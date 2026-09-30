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
    Metrics computed per measurement:
      - rate_of_change: dV/dt (units per second)
      - abs_rate_of_change: |dV/dt|
      - acceleration: d^2V/dt^2 (rate of change of derivative)
    """
    features = pd.DataFrame(index=df.index)

    if time_column in df.columns:
        dt_series = pd.to_datetime(df[time_column], utc=True).diff().dt.total_seconds()
        # Replace zero or negative dt with nominal step to avoid division by zero
        dt_series = dt_series.fillna(nominal_step_seconds)
        dt = np.where(dt_series > 0, dt_series, nominal_step_seconds)
    else:
        dt = np.full(len(df), nominal_step_seconds)

    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=np.float64)

        # First derivative: dV / dt
        diff1 = np.diff(vals, prepend=vals[0])
        roc = diff1 / dt

        # Second derivative: d(roc) / dt
        diff2 = np.diff(roc, prepend=roc[0])
        acc = diff2 / dt

        features[f"{col}_roc"] = roc
        features[f"{col}_abs_roc"] = np.abs(roc)
        features[f"{col}_acc"] = acc

    return features
