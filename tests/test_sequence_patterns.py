import numpy as np
import pandas as pd
import pytest

from scripts.experiment_sequence_patterns import PatternModel
from scripts.report_sequence_patterns import timing_control


@pytest.mark.parametrize("family", ["dictionary", "random_convolution"])
@pytest.mark.parametrize("representation", ["levels", "shape"])
def test_fixed_model_batch_prefix_and_replay(tmp_path, family, representation):
    import joblib

    rng = np.random.default_rng(42)
    values = rng.normal(size=(80, 7, 6)).astype(np.float32)
    y = np.r_[np.zeros(60, dtype=int), np.ones(20, dtype=int)]
    ids = np.r_[np.full(60, -1), np.repeat([0, 1], 10)]
    model = PatternModel(family, representation).fit(values, y, ids)
    held_out = rng.normal(size=(25, 7, 6)).astype(np.float32)
    score = model.score(held_out)
    # Later sequences cannot affect an earlier prediction through batch normalization.
    changed = held_out.copy()
    changed[8:] = 1000
    np.testing.assert_allclose(score[:8], model.score(changed)[:8], atol=1e-6)
    np.testing.assert_allclose(score[:8], model.score(held_out[:8]), atol=1e-6)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(score, joblib.load(path).score(held_out))


@pytest.mark.parametrize("alert", [False, True])
def test_timing_controls_keep_constant_alarm_segments(alert):
    t = pd.date_range("2019-01-01", periods=72, freq="5min", tz="UTC")
    frame = pd.DataFrame({"t": t, "is_alert": alert})
    result = {
        "eligible_groups": [0],
        "warned_groups": [0] if alert else [],
        "nominal_alert_fraction": float(alert),
        "events": [
            {
                "eligible": True,
                "group": 0,
                "fault_time": (t[-1] + pd.Timedelta(minutes=5)).isoformat(),
            }
        ],
    }
    control = timing_control(frame, result, repeats=20)
    assert control["segment_shift_mean"] == int(alert)
    assert control["iid_nominal_rate_expected"] == int(alert)
