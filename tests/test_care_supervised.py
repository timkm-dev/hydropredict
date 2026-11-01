import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.research_care_supervised import calibrate, train, trajectories


def test_trajectories_causal_and_gap_aware():
    times = pd.date_range("2023-01-01", periods=90, freq="10min").to_list()
    times[45:] = [t + pd.Timedelta("10min") for t in times[45:]]
    x = pd.DataFrame({"temperature": np.arange(90, dtype=float)})
    full = trajectories(times, x)
    assert full.temperature_mean36.iloc[45:80].isna().all()
    for n in [1, 35, 36, 45, 46, 80, 90]:
        pd.testing.assert_frame_equal(trajectories(times[:n], x.iloc[:n]), full.iloc[:n])


def test_held_out_turbine_cannot_change_classifier():
    rng = np.random.default_rng(42)
    cases = []
    for asset in [0, 1, 2]:
        for label in ["normal", "anomaly"]:
            cases.append(
                dict(
                    asset=asset,
                    label=label,
                    fit_idx=np.arange(100),
                    x=pd.DataFrame({"x": rng.normal(int(label == "anomaly"), 1, 100)}),
                )
            )
    with threadpool_limits(limits=2):
        first = train(cases, {2})
        for c in cases:
            if c["asset"] == 2:
                c["x"].iloc[:] = 1e9
                c["label"] = "normal"
        second = train(cases, {2})
        np.testing.assert_array_equal(
            first.predict_proba(cases[0]["x"]), second.predict_proba(cases[0]["x"])
        )


def test_threshold_fallback_is_no_alarm():
    case = dict(label="normal", times=pd.date_range("2023-01-01", periods=100, freq="10min"))
    threshold, audit = calibrate([(case, np.ones(100))])
    assert threshold == 1
    assert audit[-1]["alert_fraction"] == 0
