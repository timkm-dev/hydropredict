from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydropredict.data.preprocessing import (
    RobustTelemetryScaler,
    StandardTelemetryScaler,
    assess_sampling_cadence,
    load_scaler,
    load_telemetry_dataframe,
    resample_regular_grid,
)


def create_sample_csv(path: Path, rows: list[list[str]], header: list[str] | None = None) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["t", "V1", "V2", "V3", "V4", "V5", "V6"])
        writer.writerows(rows)
    return path


def test_load_telemetry_dataframe(tmp_path: Path) -> None:
    csv_file = tmp_path / "telemetry.csv"
    create_sample_csv(
        csv_file,
        [
            ["2026-01-01T00:05:00Z", "1.5", "2.5", "3.5", "4.5", "5.5", "6.5"],
            ["2026-01-01T00:00:00Z", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0"],
        ],
    )

    df = load_telemetry_dataframe(csv_file)
    assert len(df) == 2
    # Verify sorted chronologically
    assert df["t"].iloc[0] == pd.Timestamp("2026-01-01 00:00:00+0000")
    assert df["t"].iloc[1] == pd.Timestamp("2026-01-01 00:05:00+0000")
    assert df["V1"].dtype == np.float64


def test_load_telemetry_missing_file_or_columns(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_telemetry_dataframe(tmp_path / "nonexistent.csv")

    bad_csv = tmp_path / "bad.csv"
    create_sample_csv(bad_csv, [["2026-01-01T00:00:00Z", "1", "2"]], header=["t", "V1", "V2"])
    with pytest.raises(ValueError, match="missing required columns"):
        load_telemetry_dataframe(bad_csv)

    corrupt_csv = tmp_path / "corrupt.csv"
    create_sample_csv(
        corrupt_csv,
        [
            ["2026-01-01T00:00:00Z", "1", "2", "3", "4", "5", "6"],
            ["bad-date", "1", "2", "3", "4", "5", "6"],
        ],
    )
    with pytest.raises(ValueError, match="Validation failed"):
        load_telemetry_dataframe(corrupt_csv)

    # With strict=False, accepted rows are loaded
    df_non_strict = load_telemetry_dataframe(corrupt_csv, strict=False)
    assert len(df_non_strict) == 1


def test_assess_sampling_cadence_regular_and_gaps() -> None:
    # 5 samples spaced by 5 min (300s), plus one gap of 20 min (1200s)
    timestamps = [
        pd.Timestamp("2026-01-01 00:00:00+0000"),
        pd.Timestamp("2026-01-01 00:05:00+0000"),
        pd.Timestamp("2026-01-01 00:10:00+0000"),
        pd.Timestamp("2026-01-01 00:15:00+0000"),
        pd.Timestamp("2026-01-01 00:35:00+0000"),  # 20 min jump
        pd.Timestamp("2026-01-01 00:40:00+0000"),
    ]
    df = pd.DataFrame(
        {
            "t": timestamps,
            "V1": [1.0] * 6,
            "V2": [2.0] * 6,
            "V3": [3.0] * 6,
            "V4": [4.0] * 6,
            "V5": [5.0] * 6,
            "V6": [6.0] * 6,
        }
    )

    report = assess_sampling_cadence(df, nominal_step_seconds=300.0)
    assert report.total_records == 6
    assert report.median_step_seconds == 300.0
    assert report.gap_count == 1
    assert report.gaps[0].duration_seconds == 1200.0
    assert report.gaps[0].missing_steps_estimate == 3

    summary = report.summary()
    assert summary["gap_count"] == 1
    assert "start_time" in summary


def test_assess_sampling_cadence_too_few_rows() -> None:
    df = pd.DataFrame({"t": [pd.Timestamp("2026-01-01 00:00:00+0000")]})
    with pytest.raises(ValueError, match="At least 2 observations"):
        assess_sampling_cadence(df)


def test_resample_regular_grid_interpolates_small_gaps() -> None:
    # 00:00, 00:05, missing 00:10, 00:15, 00:20
    df = pd.DataFrame(
        {
            "t": [
                pd.Timestamp("2026-01-01 00:00:00+0000"),
                pd.Timestamp("2026-01-01 00:05:00+0000"),
                pd.Timestamp("2026-01-01 00:15:00+0000"),
                pd.Timestamp("2026-01-01 00:20:00+0000"),
            ],
            "V1": [10.0, 20.0, 40.0, 50.0],
            "V2": [1.0, 2.0, 4.0, 5.0],
            "V3": [1.0, 2.0, 4.0, 5.0],
            "V4": [1.0, 2.0, 4.0, 5.0],
            "V5": [1.0, 2.0, 4.0, 5.0],
            "V6": [1.0, 2.0, 4.0, 5.0],
        }
    )

    resampled = resample_regular_grid(df, freq="5min", max_fill_steps=2, method="linear")
    assert len(resampled) == 5
    # The 00:10 row should be linearly interpolated to 30.0
    row_10 = resampled[resampled["t"] == pd.Timestamp("2026-01-01 00:10:00+0000")].iloc[0]
    assert pytest.approx(float(row_10["V1"])) == 30.0
    assert bool(row_10["is_interpolated"]) is True

    row_05 = resampled[resampled["t"] == pd.Timestamp("2026-01-01 00:05:00+0000")].iloc[0]
    assert bool(row_05["is_interpolated"]) is False


def test_robust_telemetry_scaler_roundtrip(tmp_path: Path) -> None:
    # Create distribution with known median and IQR
    df = pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0],  # median=3, Q25=2, Q75=4, IQR=2
            "V2": [10.0, 10.0, 10.0, 10.0, 10.0],  # constant: IQR=0 -> fallback
            "V3": [100.0, 200.0, 300.0, 400.0, 500.0],
            "V4": [0.1, 0.2, 0.3, 0.4, 0.5],
            "V5": [20.0, 21.0, 22.0, 23.0, 24.0],
            "V6": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )

    scaler = RobustTelemetryScaler()
    scaled = scaler.fit_transform(df)

    # For V1: median 3, IQR 2:
    # row 0: (1 - 3) / 2 = -1.0
    # row 2: (3 - 3) / 2 = 0.0
    # row 4: (5 - 3) / 2 = 1.0
    assert pytest.approx(float(scaled["V1"].iloc[0])) == -1.0
    assert pytest.approx(float(scaled["V1"].iloc[2])) == 0.0
    assert pytest.approx(float(scaled["V1"].iloc[4])) == 1.0

    # Invert transform
    unscaled = scaler.inverse_transform(scaled)
    assert np.allclose(unscaled["V1"].to_numpy(), df["V1"].to_numpy())
    assert np.allclose(unscaled["V6"].to_numpy(), df["V6"].to_numpy())

    # Persistence
    save_path = tmp_path / "scaler.json"
    scaler.save(save_path)
    loaded_scaler = RobustTelemetryScaler.load(save_path)
    assert loaded_scaler.medians == scaler.medians
    assert loaded_scaler.iqrs == scaler.iqrs

    transformed_again = loaded_scaler.transform(df)
    assert np.allclose(transformed_again["V1"].to_numpy(), scaled["V1"].to_numpy())


def test_prepare_telemetry_series_causal_no_lookahead() -> None:
    from hydropredict.data.preprocessing import prepare_telemetry_series

    # Base series of 4 steps with 1 missing step at 00:10
    df_a = pd.DataFrame(
        {
            "t": [
                pd.Timestamp("2026-01-01 00:00:00+0000"),
                pd.Timestamp("2026-01-01 00:05:00+0000"),
                pd.Timestamp("2026-01-01 00:15:00+0000"),
            ],
            "V1": [10.0, 20.0, 100.0],
            "V2": [1.0, 2.0, 10.0],
            "V3": [1.0, 2.0, 10.0],
            "V4": [1.0, 2.0, 10.0],
            "V5": [1.0, 2.0, 10.0],
            "V6": [1.0, 2.0, 10.0],
        }
    )

    # Series B: future observation at 00:15 is perturbed from 100.0 to 9999.0
    df_b = df_a.copy()
    df_b.loc[2, "V1"] = 9999.0

    prep_a, _ = prepare_telemetry_series(df_a, method="forward")
    prep_b, _ = prepare_telemetry_series(df_b, method="forward")

    # In forward fill, value at 00:10 MUST be carried forward from 00:05 (value = 20.0)
    # Changing future value at 00:15 must NOT change the earlier value at 00:10!
    val_a_10 = float(prep_a[prep_a["t"] == pd.Timestamp("2026-01-01 00:10:00+0000")]["V1"].iloc[0])
    val_b_10 = float(prep_b[prep_b["t"] == pd.Timestamp("2026-01-01 00:10:00+0000")]["V1"].iloc[0])
    assert val_a_10 == 20.0
    assert val_b_10 == 20.0
    assert val_a_10 == val_b_10


def test_prepare_telemetry_series_outages_and_segment_boundaries() -> None:
    from hydropredict.data.preprocessing import prepare_telemetry_series

    # Telemetry with a 3-week outage between day 1 and day 21
    df = pd.DataFrame(
        {
            "t": [
                pd.Timestamp("2026-01-01 00:00:00+0000"),
                pd.Timestamp("2026-01-01 00:05:00+0000"),
                pd.Timestamp("2026-01-22 00:00:00+0000"),
                pd.Timestamp("2026-01-22 00:05:00+0000"),
            ],
            "V1": [1.0, 2.0, 3.0, 4.0],
            "V2": [1.0, 2.0, 3.0, 4.0],
            "V3": [1.0, 2.0, 3.0, 4.0],
            "V4": [1.0, 2.0, 3.0, 4.0],
            "V5": [1.0, 2.0, 3.0, 4.0],
            "V6": [1.0, 2.0, 3.0, 4.0],
        }
    )

    prep, report = prepare_telemetry_series(df, max_fill_steps=3)

    # Must NOT inflate the grid into thousands of missing rows!
    assert len(prep) == 4
    assert report.segment_count == 2
    assert report.outage_count == 1
    assert set(prep["segment_id"]) == {0, 1}
    assert prep[prep["segment_id"] == 0]["V1"].tolist() == [1.0, 2.0]
    assert prep[prep["segment_id"] == 1]["V1"].tolist() == [3.0, 4.0]


def test_gap_longer_than_max_fill_steps_not_partially_filled() -> None:
    from hydropredict.data.preprocessing import prepare_telemetry_series

    # Gap of 5 intervals (25 minutes) with max_fill_steps=3
    df = pd.DataFrame(
        {
            "t": [
                pd.Timestamp("2026-01-01 00:00:00+0000"),
                pd.Timestamp("2026-01-01 00:25:00+0000"),
            ],
            "V1": [1.0, 2.0],
            "V2": [1.0, 2.0],
            "V3": [1.0, 2.0],
            "V4": [1.0, 2.0],
            "V5": [1.0, 2.0],
            "V6": [1.0, 2.0],
        }
    )

    prep, report = prepare_telemetry_series(df, max_fill_steps=3)

    # 25 min gap > (3+1)*5min = 20min threshold -> splits into 2 segments, 0 imputed steps!
    assert report.segment_count == 2
    assert report.imputed_step_count == 0
    assert len(prep) == 2


def test_standard_telemetry_scaler_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
            "V1": [10.0, 20.0, 30.0, 40.0, 50.0],  # mean=30, std~15.811
            "V2": [5.0, 5.0, 5.0, 5.0, 5.0],  # constant: std=0 -> fallback to 1.0
            "V3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "V4": [0.1, 0.2, 0.3, 0.4, 0.5],
            "V5": [20.0, 21.0, 22.0, 23.0, 24.0],
            "V6": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )

    scaler = StandardTelemetryScaler()
    scaled = scaler.fit_transform(df)

    # For V1: mean 30, std sqrt(250) = 15.811388...
    assert pytest.approx(float(scaled["V1"].mean()), abs=1e-6) == 0.0
    assert pytest.approx(float(scaled["V1"].std()), abs=1e-6) == 1.0

    # Invert transform
    unscaled = scaler.inverse_transform(scaled)
    assert np.allclose(unscaled["V1"].to_numpy(), df["V1"].to_numpy())
    assert np.allclose(unscaled["V6"].to_numpy(), df["V6"].to_numpy())

    # Persistence roundtrip
    save_path = tmp_path / "standard_scaler.json"
    scaler.save(save_path)
    loaded = StandardTelemetryScaler.load(save_path)
    assert loaded.means == scaler.means
    assert loaded.stds == scaler.stds

    transformed_again = loaded.transform(df)
    assert np.allclose(transformed_again["V1"].to_numpy(), scaled["V1"].to_numpy())


def test_load_scaler_type_dispatch(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "V1": [1.0, 2.0, 3.0],
            "V2": [1.0, 2.0, 3.0],
            "V3": [1.0, 2.0, 3.0],
            "V4": [1.0, 2.0, 3.0],
            "V5": [1.0, 2.0, 3.0],
            "V6": [1.0, 2.0, 3.0],
        }
    )

    std_scaler = StandardTelemetryScaler().fit(df)
    std_path = tmp_path / "std.json"
    std_scaler.save(std_path)

    rob_scaler = RobustTelemetryScaler().fit(df)
    rob_path = tmp_path / "rob.json"
    rob_scaler.save(rob_path)

    loaded_std = load_scaler(std_path)
    assert isinstance(loaded_std, StandardTelemetryScaler)

    loaded_rob = load_scaler(rob_path)
    assert isinstance(loaded_rob, RobustTelemetryScaler)


def test_scaler_train_test_split_no_refit() -> None:
    df_train = pd.DataFrame(
        {
            "V1": [10.0, 20.0, 30.0],
            "V2": [1.0, 1.0, 1.0],
            "V3": [1.0, 1.0, 1.0],
            "V4": [1.0, 1.0, 1.0],
            "V5": [1.0, 1.0, 1.0],
            "V6": [1.0, 1.0, 1.0],
        }
    )
    df_test = pd.DataFrame(
        {
            "V1": [100.0, 200.0, 300.0],
            "V2": [1.0, 1.0, 1.0],
            "V3": [1.0, 1.0, 1.0],
            "V4": [1.0, 1.0, 1.0],
            "V5": [1.0, 1.0, 1.0],
            "V6": [1.0, 1.0, 1.0],
        }
    )

    scaler = StandardTelemetryScaler().fit(df_train)
    # Transforming test must use train mean (20.0) and std (10.0), NOT test statistics
    scaled_test = scaler.transform(df_test)
    assert pytest.approx(float(scaled_test["V1"].iloc[0])) == (100.0 - 20.0) / 10.0
    assert scaler.means["V1"] == 20.0


def test_scaler_unfitted_raises_runtime_error() -> None:
    scaler = StandardTelemetryScaler()
    df = pd.DataFrame({"V1": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(df)
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.inverse_transform(df)
    with pytest.raises(RuntimeError, match="Cannot serialize unfitted"):
        scaler.to_dict()
