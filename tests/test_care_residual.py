import numpy as np
import pandas as pd

from scripts.train_care_residual import context_features, fit_monitor


def test_context_is_causal_and_gap_resets():
    times = pd.date_range("2023-01-01", periods=15, freq="10min").to_list()
    times[8:] = [t + pd.Timedelta("10min") for t in times[8:]]
    values = pd.DataFrame({"x": np.arange(15, dtype=float)})
    full = context_features(times, values)
    assert full.x_mean6.iloc[5] == 2.5
    assert full.x_mean6.iloc[8:13].isna().all()
    for n in range(1, 16):
        pd.testing.assert_frame_equal(context_features(times[:n], values.iloc[:n]), full.iloc[:n])


def test_future_prediction_cannot_change_fit_or_prior_scores():
    rng = np.random.default_rng(42)
    x = rng.normal(size=6060)
    frame = pd.DataFrame(
        dict(
            time_stamp=pd.date_range("2023-01-01", periods=6060, freq="10min"),
            train_test=["train"] * 6000 + ["prediction"] * 60,
            status_type_id=0,
            context=x,
            target=2 * x + rng.normal(size=6060),
        )
    )
    first, scores, _, _, _ = fit_monitor(frame, ["context"], ["target"])
    frame.loc[6030:, "target"] = 1e6
    frame.loc[6030:, "context"] = 1e6
    frame.loc[6000:, "status_type_id"] = 4
    second, changed, _, _, _ = fit_monitor(frame, ["context"], ["target"])
    assert first["threshold"] == second["threshold"]
    np.testing.assert_allclose(scores[:30], changed[:30], atol=1e-12)
