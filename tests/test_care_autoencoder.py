import numpy as np
import pandas as pd

from scripts.select_care_autoencoder import smooth
from scripts.train_care_autoencoder import reconstruction_scores, splits, train_autoencoder


def test_splits_are_chronological_and_status_filtered():
    n = 100
    frame = pd.DataFrame(
        dict(
            train_test=["train"] * 80 + ["prediction"] * 20,
            status_type_id=[0, 2, 4, 0] * 25,
        )
    )
    valid = np.ones(n, dtype=bool)
    fit, stop, calibration, tail, pred = splits(frame, valid)
    assert fit.max() < stop.min() and stop.max() < calibration.min() < pred.min()
    assert not np.isin([2, 6, 66], np.concatenate([fit, stop, calibration])).any()
    assert tail.tolist() == list(range(64, 80)) and pred.tolist() == list(range(80, 100))


def test_smoothing_is_causal_and_keeps_unscored_readings_unscored():
    times = pd.date_range("2023-01-01", periods=12, freq="10min")
    scores = np.arange(12, dtype=float)
    scores[5] = np.nan
    result = smooth(times, scores, 3)
    assert np.isnan(result[5])
    assert result[2] == 1.0 and result[7] == 6.5
    for n in range(1, 13):
        np.testing.assert_array_equal(smooth(times[:n], scores[:n], 3), result[:n])


def test_autoencoder_scores_off_manifold_readings_higher():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(3000, 2))
    x = np.hstack([latent, latent @ rng.normal(size=(2, 6))]).astype(np.float32)
    x = (x - x.mean(0)) / x.std(0)
    model, history = train_autoencoder(x[:2500], x[2500:], epochs=15)
    assert len(history) <= 15
    broken = x[2500:].copy()
    broken[:, 3] += 4
    valid = np.ones(500, dtype=bool)
    normal = reconstruction_scores(model, x[2500:], valid)
    faulty = reconstruction_scores(model, broken, valid)
    assert np.median(faulty) > 2 * np.median(normal)
