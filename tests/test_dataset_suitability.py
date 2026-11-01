import numpy as np
import pandas as pd

from scripts.assess_dataset_suitability import CHANNELS, window_summary


def test_precursor_summary_uses_actual_time_and_excludes_fault_instant():
    cutoff = pd.Timestamp("2019-02-01", tz="UTC")
    frame = pd.DataFrame(
        {"t": pd.date_range(cutoff - pd.Timedelta(hours=6), periods=73, freq="5min")}
    )
    for channel in CHANNELS:
        frame[channel] = np.arange(73) / 12
    frame.loc[72, CHANNELS] = 1e9
    baseline, trend = window_summary(frame, cutoff, np.ones(6))
    np.testing.assert_allclose(trend, 6)
    np.testing.assert_allclose(baseline, np.mean(np.arange(12) / 12))
    assert window_summary(frame.iloc[20:], cutoff, np.ones(6)) is None
    assert window_summary(frame.iloc[:-4], cutoff, np.ones(6)) is None
