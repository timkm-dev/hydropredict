from __future__ import annotations

import pandas as pd
import pytest

from hydropredict.analysis.readiness import event_metrics, fault_window_coverage, nominal_metrics


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


def test_event_and_nominal_metrics_use_six_hour_windows_and_gap_episodes():
    t = pd.date_range("2019-03-01", periods=8, freq="5min", tz="UTC")
    scores = pd.DataFrame(
        {"t": t, "anomaly_score": [1, 5, 1, 1, 7, 1, 1, 1], "is_alert": [0, 1, 0, 0, 1, 0, 0, 0]}
    ).astype({"is_alert": bool})
    fault = t[-1] + pd.Timedelta(minutes=5)
    late = t[0] + pd.Timedelta(hours=7)
    coverage = [
        {"fault_time": fault.isoformat(), "eligible": True},
        {"fault_time": late.isoformat(), "eligible": True},
        {"fault_time": fault.isoformat(), "eligible": False},
    ]
    result = event_metrics(scores, coverage)
    assert (result["eligible_events"], result["detected_events"], result["recall"]) == (2, 1, 0.5)
    assert result["events"][0]["lead_minutes"] == 35 and result["events"][0]["max_score"] == 7
    assert result["events"][1]["detected"] is False
    nominal = nominal_metrics(scores)
    assert nominal["alert_episodes"] == 2 and nominal["point_alerts"] == 2
    assert nominal["observed_hours"] == 35 / 60
