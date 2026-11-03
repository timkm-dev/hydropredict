import numpy as np
import pandas as pd

from scripts.train_care_autoencoder import train_autoencoder
from scripts.train_care_autoencoder_v2 import CLIP, inputs, scores, standardise


def frame(n=200):
    rng = np.random.default_rng(0)
    ambient = rng.normal(20, 5, n)
    return pd.DataFrame(
        dict(
            sensor_0_avg=ambient,
            sensor_1_avg=ambient + 30 + rng.normal(0, 1, n),
            sensor_2_avg=np.where(np.arange(n) % 50 == 0, 26.001, 26.0),
            sensor_3_avg=rng.normal(0, 1, n),
        )
    )


DESCRIPTIONS = pd.Series(
    {
        "sensor_0": "Ambient temperature",
        "sensor_1": "Gearbox oil temperature",
        "sensor_2": "24VDC power pack",
        "sensor_3": "Rotor speed",
    }
)


def test_quasi_constant_channels_are_dropped():
    x = inputs(frame(), np.arange(150), DESCRIPTIONS, "sensor_0_avg", delta=False)
    assert list(x.columns) == ["sensor_0_avg", "sensor_1_avg", "sensor_3_avg"]


def test_temperatures_become_above_ambient_but_ambient_stays():
    raw = frame()
    x = inputs(raw, np.arange(150), DESCRIPTIONS, "sensor_0_avg", delta=True)
    np.testing.assert_allclose(x.sensor_1_avg, raw.sensor_1_avg - raw.sensor_0_avg)
    np.testing.assert_array_equal(x.sensor_0_avg, raw.sensor_0_avg)
    np.testing.assert_array_equal(x.sensor_3_avg, raw.sensor_3_avg)


def test_standardised_inputs_are_clipped_and_missing_becomes_the_fit_mean():
    x = frame()[["sensor_1_avg", "sensor_3_avg"]].copy()
    x.iloc[180, 0] = 1e9
    x.iloc[181, 1] = np.nan
    z, _, _ = standardise(x, np.arange(150))
    assert z[180, 0] == CLIP and np.abs(z).max() <= CLIP
    assert z[181, 1] == 0


def test_per_sensor_score_is_not_dominated_by_one_noisy_channel():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(3000, 2))
    x = np.hstack([latent, latent @ rng.normal(size=(2, 6))]).astype(np.float32)
    x = (x - x.mean(0)) / x.std(0)
    x[:, 7] += rng.normal(0, 3, len(x)).astype(np.float32)  # naturally erratic channel
    model, _ = train_autoencoder(x[:2000], x[2000:2500], epochs=15)
    shifted = x.copy()
    shifted[2500:, 2] += 3  # a quiet channel drifts
    rows = np.arange(2000, 3000)
    valid = np.ones(500, dtype=bool)
    plain, per_sensor = scores(model, shifted, rows, 500, valid)
    plain0, per_sensor0 = scores(model, x, rows, 500, valid)
    # The quiet channel's drift stands out more once each channel is judged on its own scale.
    assert np.median(per_sensor / per_sensor0) > np.median(plain / plain0)
