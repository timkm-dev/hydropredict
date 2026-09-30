"""Rolling window temporal feature extraction for hydroelectric telemetry."""

from __future__ import annotations

import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS


def compute_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (6, 24),
    columns: tuple[str, ...] = MEASUREMENT_COLUMNS,
) -> pd.DataFrame:
    """Compute backward-looking rolling statistics across specified window sizes.

    For a 5-minute sampling cadence:
      - window=6 corresponds to 30 minutes.
      - window=24 corresponds to 2 hours.

    Computed metrics for each measurement and window:
      - rolling_mean: Central tendency
      - rolling_std: Local volatility / noise
      - rolling_min: Local floor
      - rolling_max: Local ceiling
      - rolling_p2p: Local peak-to-peak amplitude (max - min)
    """
    features = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]

        for w in windows:
            roll = series.rolling(window=w, min_periods=1)
            mean_series = roll.mean()
            std_series = roll.std().fillna(0.0)
            min_series = roll.min()
            max_series = roll.max()
            p2p_series = max_series - min_series

            features[f"{col}_roll_mean_{w}"] = mean_series
            features[f"{col}_roll_std_{w}"] = std_series
            features[f"{col}_roll_min_{w}"] = min_series
            features[f"{col}_roll_max_{w}"] = max_series
            features[f"{col}_roll_p2p_{w}"] = p2p_series

    return features
