"""Checks for physical time, causality and unavailable history in trend scoring."""

import numpy as np
import pandas as pd
import pytest

from hydropredict.features.trends import trend_features
from scripts.evaluate_normal_trends import persistence


def test_elapsed_hour_slope_and_prefix_invariance():
    # Irregular cadence: row-index regression would produce a different slope.
    seconds = np.cumsum(np.tile([240, 360], 60))
    times = pd.Timestamp("2019-01-01", tz="UTC") + pd.to_timedelta(seconds, unit="s")
    frame = pd.DataFrame({"t": times, "V1": seconds / 3600 * 2 + 7})
    result = trend_features(frame, ["V1"])
    assert len(result)
    np.testing.assert_allclose(result.V1_slope_per_hour, 2)
    prefix = trend_features(frame.iloc[:100], ["V1"])
    pd.testing.assert_frame_equal(prefix, result[result.t.isin(prefix.t)].reset_index(drop=True))


def test_outage_requires_new_six_hour_warmup():
    times = pd.date_range("2019-01-01", periods=100, freq="5min", tz="UTC")
    frame = pd.DataFrame({"t": times, "V1": np.arange(100)})
    frame.loc[80:, "t"] += pd.Timedelta(hours=2)
    result = trend_features(frame, ["V1"])
    assert result.t.max() == times[79]


def test_reject_unsorted_input():
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2019-01-01", periods=80, freq="5min", tz="UTC"),
            "V1": np.arange(80),
        }
    )
    with pytest.raises(ValueError):
        trend_features(frame.iloc[::-1], ["V1"])


def test_persistence_restarts_after_unavailable_history():
    times = pd.Series(
        pd.to_datetime(
            [
                "2019-01-01 00:00Z",
                "2019-01-01 00:05Z",
                "2019-01-01 00:10Z",
                "2019-01-01 01:00Z",
                "2019-01-01 01:05Z",
                "2019-01-01 01:10Z",
            ]
        )
    )
    above = pd.Series([True] * 6)
    assert persistence(times, above).tolist() == [False, False, True, False, False, True]
