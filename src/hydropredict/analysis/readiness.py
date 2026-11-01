"""Retrospective fault-window coverage, independent of model fitting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def fault_window_coverage(
    timestamps: pd.Series[Any],
    faults: Sequence[pd.Timestamp],
    *,
    warning_hours: float = 6.0,
    cadence_seconds: int = 300,
    minimum_coverage: float = 0.8,
    maximum_last_gap_seconds: int = 600,
) -> list[dict[str, Any]]:
    """Require 80% of pre-fault slots and an observation within ten minutes by default.

    Slots are anchored to the fault, and duplicate observations cannot inflate coverage.
    These are declared research assumptions, not externally validated standards.
    """
    if warning_hours <= 0 or cadence_seconds <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("Invalid fault coverage configuration.")
    times = pd.to_datetime(timestamps, utc=True)
    if times.isna().any():
        raise ValueError("Coverage timestamps must be valid.")
    duration = warning_hours * 3600
    expected = int(np.ceil(duration / cadence_seconds))
    result: list[dict[str, Any]] = []
    for fault in faults:
        seconds = (fault - times).dt.total_seconds()
        prior = seconds[(seconds > 0) & (seconds <= duration)]
        occupied = len(np.unique(np.floor((prior.to_numpy() - 1e-9) / cadence_seconds)))
        coverage = occupied / expected
        last_gap = float(prior.min()) if not prior.empty else None
        result.append(
            {
                "fault_time": fault.isoformat(),
                "observations": len(prior),
                "occupied_slots": occupied,
                "expected_slots": expected,
                "coverage_fraction": coverage,
                "last_observation_gap_seconds": last_gap,
                "has_any_prior_data": bool(len(prior)),
                "eligible": coverage >= minimum_coverage
                and last_gap is not None
                and last_gap <= maximum_last_gap_seconds,
            }
        )
    return result


def event_metrics(scores: pd.DataFrame, coverage: list[dict[str, Any]]) -> dict[str, Any]:
    """Alerts in the six hours before each fault: detection, lead time and peak score."""
    alert_times = pd.to_datetime(scores.loc[scores.is_alert, "t"], utc=True)
    events = []
    for row in coverage:
        event = dict(row)
        fault = pd.Timestamp(event["fault_time"])
        preceding = alert_times[
            (alert_times < fault) & (alert_times >= fault - pd.Timedelta(hours=6))
        ]
        event["detected"] = bool(len(preceding))
        event["lead_minutes"] = (
            (fault - preceding.min()).total_seconds() / 60 if len(preceding) else None
        )
        window = scores[(scores.t < fault) & (scores.t >= fault - pd.Timedelta(hours=6))]
        event["max_score"] = float(window.anomaly_score.max()) if len(window) else None
        events.append(event)
    eligible = [e for e in events if e["eligible"]]
    detected = [e for e in eligible if e["detected"]]
    return {
        "eligible_events": len(eligible),
        "detected_events": len(detected),
        "recall": len(detected) / len(eligible) if eligible else None,
        "median_lead_minutes": float(np.median([e["lead_minutes"] for e in detected]))
        if detected
        else None,
        "events": events,
    }


def nominal_metrics(scores: pd.DataFrame) -> dict[str, Any]:
    """Alert burden on normal-reference readings; gaps over 450 s start new episodes."""
    gaps = scores.t.diff().dt.total_seconds()
    new_episode = ~scores.is_alert.shift(fill_value=False) | gaps.gt(450)
    seconds = float(gaps[(gaps > 0) & (gaps <= 450)].sum())
    episodes = int((scores.is_alert & new_episode).sum())
    return {
        "point_alert_fraction": float(scores.is_alert.mean()),
        "point_alerts": int(scores.is_alert.sum()),
        "observations": len(scores),
        "alert_episodes": episodes,
        "observed_hours": seconds / 3600,
        "episodes_per_observed_day": episodes / (seconds / 86400) if seconds else None,
    }
