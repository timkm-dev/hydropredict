import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

from hydropredict.analysis.care_score import (
    care_score,
    criticality,
    earliness_weights,
    f_beta,
    score_event,
)


def reference_weighted_score(prediction, numerator=1, denominator=4):
    """Line-for-line copy of the reference _calculate_weighted_score arithmetic."""
    start = numerator / denominator
    scale = int(denominator)
    scaled = len(prediction) * scale
    cp = int(scaled * start)
    x = np.linspace(0, scale, scaled)
    weights = np.zeros(scaled)
    weights[:cp] = scale
    weights[cp:scaled] = scale / (1 - start) - 1 / (1 - start) * x[cp:scaled]
    weights = weights[::scale] / scale
    return np.sum(prediction * weights) / np.sum(weights)


def test_f_beta_matches_scikit_learn():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, p = rng.random(40) < 0.4, rng.random(40) < 0.3
        tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
        expected = fbeta_score(y, p, beta=0.5, zero_division=0)
        assert f_beta(tp, fp, fn, 0.5) == pytest.approx(expected)


def test_criticality_counts_only_normal_status_and_floors_at_zero():
    predicted = np.array([1, 1, 0, 1, 1, 0, 0, 0], dtype=bool)
    normal = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert criticality(predicted, normal).tolist() == [1, 2, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("length", [1, 7, 100, 1009])
@pytest.mark.parametrize("descend", [(1, 4), (1, 2)])
def test_earliness_weights_match_reference_arithmetic(length, descend):
    rng = np.random.default_rng(length)
    prediction = rng.random(length) < 0.5
    weights = earliness_weights(length, descend)
    ours = np.sum(prediction * weights) / np.sum(weights)
    assert ours == pytest.approx(reference_weighted_score(prediction, *descend))
    assert weights[0] == 1 and weights.min() >= 0


def test_early_detection_scores_higher_than_late_detection():
    times = pd.date_range("2023-01-01", periods=400, freq="10min")
    normal = np.ones(400, dtype=bool)
    start, end = times[100], times[399]
    early, late = np.zeros(400, dtype=bool), np.zeros(400, dtype=bool)
    early[100:200], late[300:400] = True, True
    a = score_event(1, "anomaly", times, early, normal, start, end)
    b = score_event(2, "anomaly", times, late, normal, start, end)
    assert a.weighted_score > b.weighted_score
    assert a.max_criticality == 100 and a.first_detection == times[171]
    assert (a.tp, a.fp, a.fn, a.tn) == (100, 0, 200, 100)


def test_abnormal_status_rows_are_excluded_from_point_metrics():
    times = pd.date_range("2023-01-01", periods=10, freq="10min")
    predicted = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    normal = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    event = score_event(1, "normal", times, predicted, normal, times[0], times[-1])
    assert event.accuracy == 1.0 and event.fp == 0 and event.max_criticality == 0


def _events(detect_anomaly, normal_alarm_rows):
    times = pd.date_range("2023-01-01", periods=200, freq="10min")
    ok = np.ones(200, dtype=bool)
    anomaly = np.zeros(200, dtype=bool)
    if detect_anomaly:
        anomaly[100:190] = True
    quiet = np.zeros(200, dtype=bool)
    quiet[:normal_alarm_rows] = True
    return [
        score_event(1, "anomaly", times, anomaly, ok, times[100], times[199]),
        score_event(2, "normal", times, quiet, ok, times[0], times[199]),
    ]


def test_care_is_zero_without_any_detection():
    assert care_score(_events(False, 0))["care"] == 0.0


def test_care_falls_back_to_accuracy_when_normal_accuracy_is_poor():
    result = care_score(_events(True, 150))
    assert result["accuracy"] == 0.25 and result["care"] == 0.25


def test_care_weights_accuracy_twice():
    result = care_score(_events(True, 0))
    expected = (
        result["coverage"] + result["earliness"] + result["reliability"] + 2 * result["accuracy"]
    ) / 5
    assert result["reliability"] == 1.0 and result["care"] == pytest.approx(expected)


def test_care_requires_both_labels():
    with pytest.raises(ValueError):
        care_score(_events(True, 0)[:1])
