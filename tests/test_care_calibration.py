import numpy as np
import pandas as pd

from scripts.check_care_model_calibration import calibrate_quantiles, native_scores
from scripts.compare_care_models import make_model


def test_quantile_calibration_invariant_to_affine_score_scale():
    rng = np.random.default_rng(42)
    case = dict(label="normal", times=pd.date_range("2023-01-01", periods=10000, freq="10min"))
    scores = rng.normal(size=10000)
    first, q1, _ = calibrate_quantiles([(case, scores)])
    second, q2, _ = calibrate_quantiles([(case, scores * 37 + 100)])
    assert q1 == q2
    np.testing.assert_array_equal(scores > first, scores * 37 + 100 > second)


def test_naive_bayes_log_odds_preserve_saturated_score_order():
    x = pd.DataFrame({"x": [-1.1, -1, -0.9, 0.9, 1, 1.1]})
    model = make_model("gaussian_nb", 1)
    model.fit(x, [0, 0, 0, 1, 1, 1])
    target = pd.DataFrame({"x": [5.0, 10.0]})
    probability = model.predict_proba(target)[:, 1]
    scores = native_scores(model, target, "gaussian_nb")
    assert probability[0] == probability[1] == 1
    assert np.isfinite(scores).all() and scores[1] > scores[0]
