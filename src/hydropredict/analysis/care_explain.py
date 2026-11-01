"""Per-sensor explanations of autoencoder alerts; see docs/care-explanation-protocol.md."""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
import pandas as pd

FAULT_COMPONENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("transformer", ("transformer",)),
    ("hydraulic", ("hydraulic",)),
    ("gearbox", ("gearbox",)),
    ("generator", ("generator bearing",)),
    ("rotor bearing", ("rotor bearing", "rotorbearing", "main bearing")),
)


def fault_component(description: str) -> str | None:
    """Component named by a fault description, by the first matching keyword group."""
    text = description.lower()
    for component, keywords in FAULT_COMPONENTS:
        if any(k in text for k in keywords):
            return component
    return None


def sensor_component(description: str) -> str | None:
    """Component a sensor monitors, from its description."""
    text = description.lower()
    if "transformer" in text or "transformator" in text:
        return "transformer"
    if "hydraulic" in text:
        return "hydraulic"
    if "gearbox" in text:
        return "gearbox"
    if "generator" in text and "temperature" in text:
        return "generator"
    if "rotor bearing" in text:
        return "rotor bearing"
    return None


def contribution_shares(
    times: pd.DatetimeIndex, errors: np.ndarray[Any, Any], names: list[str], window: int
) -> pd.DataFrame:
    """Causal smoothed per-sensor squared errors as shares of their total.

    ``errors`` holds squared standardised reconstruction errors (rows x sensors);
    rows that are NaN stay NaN, matching unscored readings.
    """
    frame = pd.DataFrame(errors, index=times, columns=names)
    smoothed = frame.rolling(pd.Timedelta(minutes=10 * window), min_periods=1).mean()
    smoothed[frame.isna().all(axis=1)] = np.nan
    return smoothed.div(smoothed.sum(axis=1), axis=0)


def top_sensors(shares: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    best = shares.dropna().sort_values(ascending=False, kind="stable").head(k)
    return [(str(name), float(value)) for name, value in best.items()]


def chance_of_hit(sensors: int, matching: int, k: int = 3) -> float:
    """Probability that k sensors drawn without replacement include a match."""
    k = min(k, sensors)
    return 1 - comb(sensors - matching, k) / comb(sensors, k)


def at_least(probabilities: list[float], observed: int) -> float:
    """Exact P(at least ``observed`` successes) for independent Bernoulli trials."""
    dist = np.zeros(len(probabilities) + 1)
    dist[0] = 1.0
    for p in probabilities:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[observed:].sum())
