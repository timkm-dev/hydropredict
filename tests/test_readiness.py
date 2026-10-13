from __future__ import annotations

import pandas as pd
import pytest

from hydropredict.analysis.readiness import fault_window_coverage


def test_coverage_excludes_fault_time_and_future_and_deduplicates() -> None:
    fault = pd.Timestamp("2026-01-01T06:00:00Z")
    times = pd.Series(pd.date_range("2026-01-01", periods=73, freq="5min", tz="UTC"))
    result = fault_window_coverage(pd.concat([times, times]), [fault])[0]
    assert result["eligible"]
    assert result["occupied_slots"] == 72
    assert result["coverage_fraction"] == 1
    assert not fault_window_coverage(pd.Series([fault]), [fault])[0]["eligible"]


def test_sparse_or_stale_coverage_does_not_qualify() -> None:
    fault = pd.Timestamp("2026-01-01T06:00:00Z")
    sparse = pd.Series([fault - pd.Timedelta(minutes=5)])
    assert not fault_window_coverage(sparse, [fault])[0]["eligible"]
    stale = pd.Series(pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC"))
    assert not fault_window_coverage(stale, [fault])[0]["eligible"]
    with pytest.raises(ValueError):
        fault_window_coverage(sparse, [fault], warning_hours=0)
