"""Verify trend calculations cannot turn unavailable telemetry into evidence."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_signal_overlap import read_scores, six_hour_slope


def test_loads_score_times_with_mixed_fractional_precision(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("t\n2019-02-02 05:42:46.123000+00:00\n2019-02-02 05:47:46+00:00\n")
    frame = read_scores(path)
    assert len(frame) == 2
    assert str(frame.t.dt.tz) == "UTC"


def test_known_slope_and_coverage_exclusions() -> None:
    fault = pd.Timestamp("2020-01-02", tz="UTC")
    times = pd.date_range(fault - pd.Timedelta(hours=6), periods=72, freq="5min")
    frame = pd.DataFrame({"t": times})
    for index in range(1, 7):
        frame[f"V{index}"] = np.arange(72) / 12 * index
    slope = six_hour_slope(frame, fault, np.ones(6))
    np.testing.assert_allclose(slope, np.arange(1, 7) * 6)
    assert six_hour_slope(frame.iloc[:30], fault, np.ones(6)) is None
    assert six_hour_slope(frame.iloc[:-3], fault, np.ones(6)) is None
    assert six_hour_slope(frame.iloc[:0], fault, np.ones(6)) is None
    # The strict pre-fault interval never includes a future extreme value.
    future = frame.iloc[[-1]].copy()
    future["t"] = fault
    future["V1"] = 1e6
    np.testing.assert_allclose(six_hour_slope(pd.concat([frame, future]), fault, np.ones(6)), slope)
