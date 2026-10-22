"""Domain-specific engineering features for small hydroelectric plant (SHP) telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_domain_features(
    df: pd.DataFrame,
    epsilon: float = 1e-6,
    min_power_floor: float = 50.0,
) -> pd.DataFrame:
    """Compute physical and thermodynamic domain indicators from telemetry in engineering units.

    Sensor channel semantics in the SHP benchmark:
      - V1, V2: Generator guide bearing horizontal/vertical radial vibration (mm/s).
      - V3: Turbine guide bearing radial vibration (mm/s).
      - V4: Thrust bearing axial vibration (mm/s).
      - V5: Bearing lubrication operating temperature (°C).
      - V6: Generator active electrical power output (kW).

    Indicators:
      - vib_total_rss: Root-sum-of-squares across all 4 vibration channels (aggregate intensity).
      - vib_bearing_balance: Ratio of generator guide to turbine guide bearing vibrations.
      - vib_crest_factor: Cross-channel peak-to-RSS proxy ratio (max(V1..V4) / vib_total_rss).
        Note: This is a spatial cross-channel proxy ratio across concurrent RMS channels,
        not a conventional single-waveform time-domain peak-to-RMS crest factor.
      - vib_power_intensity: Vibration per unit electrical power (degradation indicator).
      - thermal_power_intensity: Bearing temperature per unit electrical power.

    The `min_power_floor` parameter (default 50.0 kW) is a configurable engineering
    regularization floor preventing division by zero during machine shutdown, tailwater
    depression, or low-load spinning reserve regimes.
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

        # Root sum of squares across 4 vibration channels
        vib_rss = np.sqrt(v1**2 + v2**2 + v3**2 + v4**2)
        features["vib_total_rss"] = vib_rss

        # Guide bearing spatial balance (generator vs turbine)
        features["vib_bearing_balance"] = (v1 + v2 + epsilon) / (v3 + v4 + epsilon)

        # Spatial peak vibration proxy across channels relative to total RSS
        vib_max = np.maximum(np.maximum(v1, v2), np.maximum(v3, v4))
        features["vib_crest_factor"] = vib_max / (vib_rss + epsilon)

        if has_power:
            power = np.maximum(df["V6"].to_numpy(dtype=np.float64), min_power_floor)
            features["vib_power_intensity"] = vib_rss / power

    if has_temp and has_power:
        temp = df["V5"].to_numpy(dtype=np.float64)
        power = np.maximum(df["V6"].to_numpy(dtype=np.float64), min_power_floor)
        features["thermal_power_intensity"] = temp / power

    return features
