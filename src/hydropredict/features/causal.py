"""Time-based numerical features with explicit causal warm-up and outage resets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS


def causal_feature_frame(
    frame: pd.DataFrame, mode: str = "raw", context_minutes: int = 120
) -> pd.DataFrame:
    """Use only available readings; never interpolate or bridge gaps over 7.5 minutes.

    Temporal output requires two hours of context after each reset. Caller retains
    coverage exclusions. Each call starts fresh; streaming callers must supply history.
    """
    if mode not in ("raw", "temporal"):
        raise ValueError(f"Unsupported feature mode: {mode}")
    window_sets = {
        15: ((10, 2), (15, 2)),
        30: ((15, 2), (30, 3)),
        120: ((30, 3), (120, 12)),
    }
    if context_minutes not in window_sets:
        raise ValueError("Context must be 15, 30 or 120 minutes.")
    times = pd.to_datetime(frame["t"], utc=True)
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be valid, unique and chronological.")
    values = frame[list(MEASUREMENT_COLUMNS)].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Feature measurements must be finite.")
    result = values.copy()
    if mode == "raw":
        result.insert(0, "t", times)
        return result
    segments = (times.diff().dt.total_seconds().gt(450) | times.diff().isna()).cumsum()
    parts = []
    for positions in frame.groupby(segments, sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        group = values.iloc[positions].copy()
        group_times = times.iloc[positions]
        group.index = pd.DatetimeIndex(group_times)
        engineered = group.copy()
        for minutes, minimum in window_sets[context_minutes]:
            rolling = group.rolling(f"{minutes}min", min_periods=minimum)
            for statistic, derived in (("mean", rolling.mean()), ("std", rolling.std())):
                for col in MEASUREMENT_COLUMNS:
                    engineered[f"{col}_{statistic}_{minutes}min"] = derived[col]
        dt = group.index.to_series().diff().dt.total_seconds()
        for col in MEASUREMENT_COLUMNS:
            engineered[f"{col}_roc"] = group[col].diff() / dt
        eligible = group.index >= group.index[0] + pd.Timedelta(minutes=context_minutes)
        engineered = engineered.loc[eligible].dropna()
        engineered.insert(0, "t", engineered.index)
        parts.append(engineered.reset_index(drop=True))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
