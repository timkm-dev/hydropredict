import numpy as np
import pandas as pd

from scripts.train_care_baseline import episode_starts, persistent_alerts, run_file


def test_alerts_are_causal_and_reset_after_gap():
    times = pd.date_range("2023-01-01", periods=14, freq="10min").to_list()
    times[8:] = [t + pd.Timedelta("10min") for t in times[8:]]
    exceed = np.ones(14, dtype=bool)
    result = persistent_alerts(times, exceed)
    assert np.flatnonzero(result).tolist() == [5, 6, 7, 13]
    for n in range(1, 15):
        np.testing.assert_array_equal(persistent_alerts(times[:n], exceed[:n]), result[:n])
    assert episode_starts(times, result).sum() == 2


def test_nonexceedance_restarts_persistence():
    times = pd.date_range("2023-01-01", periods=12, freq="10min")
    exceed = np.ones(12, dtype=bool)
    exceed[5] = False
    assert np.flatnonzero(persistent_alerts(times, exceed)).tolist() == [11]


def test_prediction_values_and_status_cannot_change_fitted_threshold(tmp_path):
    folder = tmp_path / "Farm" / "datasets"
    folder.mkdir(parents=True)
    models = tmp_path / "models"
    models.mkdir()
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01", periods=6060, freq="10min")
    frame = pd.DataFrame(
        dict(
            time_stamp=times,
            asset_id=0,
            train_test=["train"] * 6000 + ["prediction"] * 60,
            status_type_id=0,
            sensor_0_avg=rng.normal(size=6060),
        )
    )
    event = pd.Series(
        dict(
            event_start=times[6000],
            event_end=times[-1],
            event_label="anomaly",
            event_description="synthetic",
        )
    )
    path = folder / "1.csv"
    frame.to_csv(path, sep=";", index=False)
    first = run_file(path, event, ["sensor_0_avg"], models)
    before = pd.read_csv(models / "Farm_1_scores.csv")
    frame.loc[6030:, "sensor_0_avg"] = 1e6
    frame.loc[6000:, "status_type_id"] = 4
    frame.to_csv(path, sep=";", index=False)
    second = run_file(path, event, ["sensor_0_avg"], models)
    after = pd.read_csv(models / "Farm_1_scores.csv")
    assert first["threshold"] == second["threshold"]
    np.testing.assert_array_equal(before.score[:30], after.score[:30])
    np.testing.assert_array_equal(before.alert[:30], after.alert[:30])
