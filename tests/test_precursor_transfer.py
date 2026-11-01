"""Relative features must be causal; selection cannot use a failing fold."""

import numpy as np
import pandas as pd

from scripts.backtest_precursor_transfer import choose, relative_features


def test_relative_features_ignore_level_and_keep_elapsed_time_causality():
    times = pd.date_range("2018-08-01", periods=40, freq="5min", tz="UTC")
    hours = np.arange(40) / 12
    frame = pd.DataFrame({"t": times, **{f"V{i}": hours * i + 20 for i in range(1, 7)}})
    output = relative_features(frame)
    assert np.allclose(output.V3_slope, 3)
    prefix = relative_features(frame.iloc[:20])
    pd.testing.assert_frame_equal(prefix, output[output.t.le(times[19])].reset_index(drop=True))
    shifted = frame.copy()
    shifted.loc[:, "V1":"V6"] += 100
    np.testing.assert_allclose(
        relative_features(shifted).drop(columns="t"), output.drop(columns="t"), atol=1e-8
    )
    broken = frame.copy()
    broken.loc[20:, "t"] += pd.Timedelta(hours=1)
    assert not relative_features(broken).t.between(broken.t.iloc[20], broken.t.iloc[25]).any()


def result(name, warned, rate):
    return dict(
        candidate=name,
        metrics=dict(
            warned_groups=warned,
            nominal_alert_fraction=rate,
            nominal_alerts=int(rate * 10000),
            nominal_points=10000,
        ),
    )


def test_selection_requires_both_folds_and_budget_then_breaks_ties():
    results = [
        result("absolute_hgb", [1], 0.01),
        result("absolute_hgb", [], 0.01),
        result("relative_hgb", [1, 2], 0.03),
        result("relative_hgb", [1], 0.01),
    ]
    assert choose(results) is None
    results += [
        result("absolute_logistic", [1], 0.01),
        result("absolute_logistic", [1], 0.01),
        result("relative_logistic", [1], 0.005),
        result("relative_logistic", [1], 0.005),
    ]
    assert choose(results) == "relative_logistic"
