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
