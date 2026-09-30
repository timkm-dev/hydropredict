"""Domain-specific engineering features for small hydroelectric plant (SHP) telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_domain_features(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """Compute physical and thermodynamic domain features.

    Variables in the SHP benchmark:
      - V1, V2, V3, V4: Bearing and shaft vibration sensors.
      - V5: Bearing / oil operating temperature.
      - V6: Generator active electrical power output (kW).

    Domain indicators:
      - vib_total_rms: Root-sum-of-squares across all 4 vibration channels.
      - vib_power_intensity: Vibration per unit of electrical power (degradation indicator).
      - vib_radial_axial_balance: Relative balance between turbine and generator guide bearings.
      - thermal_power_intensity: Operating temperature per unit of electrical power.
      - vib_crest_factor: Ratio of maximum single-channel vibration to aggregate RMS.
    """
    features = pd.DataFrame(index=df.index)

    has_vib = all(col in df.columns for col in ("V1", "V2", "V3", "V4"))
    has_temp = "V5" in df.columns
    has_power = "V6" in df.columns

    if has_vib:
        v1 = df["V1"].to_numpy(dtype=np.float64)
        v2 = df["V2"].to_numpy(dtype=np.float64)
        v3 = df["V3"].to_numpy(dtype=np.float64)
        v4 = df["V4"].to_numpy(dtype=np.float64)

        # Root sum of squares
        vib_rss = np.sqrt(v1**2 + v2**2 + v3**2 + v4**2)
        features["vib_total_rss"] = vib_rss

        # Guide bearing balance
        features["vib_bearing_balance"] = (v1 + v2 + epsilon) / (v3 + v4 + epsilon)

        # Peak vibration across channels
        vib_max = np.maximum(np.maximum(v1, v2), np.maximum(v3, v4))
        features["vib_crest_factor"] = vib_max / (vib_rss + epsilon)

        if has_power:
            power = np.maximum(
                df["V6"].to_numpy(dtype=np.float64), 50.0
            )  # clamp to avoid low-load div
            features["vib_power_intensity"] = vib_rss / power

    if has_temp and has_power:
        temp = df["V5"].to_numpy(dtype=np.float64)
        power = np.maximum(df["V6"].to_numpy(dtype=np.float64), 50.0)
        features["thermal_power_intensity"] = temp / power

    return features
