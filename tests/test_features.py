from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydropredict.features import (
    FeatureConfig,
    compute_domain_features,
    compute_rolling_features,
    compute_transient_features,
    extract_feature_matrix,
)


@pytest.fixture
def sample_telemetry() -> pd.DataFrame:
    # 10 rows spaced 300 seconds apart
    times = pd.date_range("2026-01-01 00:00:00", periods=10, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "t": times,
            "V1": [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0],
            "V2": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
            "V3": [0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 1.2, 1.2, 1.5, 1.5],
            "V4": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3],
            "V5": [18.0, 18.2, 18.5, 18.7, 19.0, 19.2, 19.5, 19.8, 20.0, 20.2],
            "V6": [3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0],
        }
    )


def test_rolling_features(sample_telemetry: pd.DataFrame) -> None:
    features = compute_rolling_features(sample_telemetry, windows=(3,), columns=("V1",))

    assert "V1_roll_mean_3" in features.columns
    assert "V1_roll_std_3" in features.columns
    assert "V1_roll_min_3" in features.columns
    assert "V1_roll_max_3" in features.columns
    assert "V1_roll_p2p_3" in features.columns

    # First row window=3 has min_periods=1 -> mean is 0.2, std is 0.0, p2p is 0.0
    assert features["V1_roll_mean_3"].iloc[0] == pytest.approx(0.2)
    assert features["V1_roll_p2p_3"].iloc[0] == pytest.approx(0.0)

    # Row 2 (index 2: values 0.2, 0.2, 0.4) -> mean = 0.8/3, min = 0.2, max = 0.4, p2p = 0.2
    assert features["V1_roll_mean_3"].iloc[2] == pytest.approx(0.8 / 3)
    assert features["V1_roll_p2p_3"].iloc[2] == pytest.approx(0.2)


def test_transient_features(sample_telemetry: pd.DataFrame) -> None:
    transients = compute_transient_features(
        sample_telemetry, columns=("V1",), nominal_step_seconds=300.0
    )

    assert "V1_roc" in transients.columns
    assert "V1_abs_roc" in transients.columns
    assert "V1_acc" in transients.columns

    # First row delta is 0
    assert transients["V1_roc"].iloc[0] == pytest.approx(0.0)

    # Between row 1 (0.2) and row 2 (0.4), delta = 0.2 over 300s -> roc = 0.2 / 300
    expected_roc = 0.2 / 300.0
    assert transients["V1_roc"].iloc[2] == pytest.approx(expected_roc)
    assert transients["V1_abs_roc"].iloc[2] == pytest.approx(expected_roc)


def test_domain_features(sample_telemetry: pd.DataFrame) -> None:
    domain = compute_domain_features(sample_telemetry)

    assert "vib_total_rss" in domain.columns
    assert "vib_bearing_balance" in domain.columns
    assert "vib_crest_factor" in domain.columns
    assert "vib_power_intensity" in domain.columns
    assert "thermal_power_intensity" in domain.columns

    # Row 0: V1=0.2, V2=0.1, V3=0.3, V4=0.1
    # rss = sqrt(0.04 + 0.01 + 0.09 + 0.01) = sqrt(0.15)
    expected_rss = np.sqrt(0.2**2 + 0.1**2 + 0.3**2 + 0.1**2)
    assert domain["vib_total_rss"].iloc[0] == pytest.approx(expected_rss)

    # Power intensity: rss / 3000
    assert domain["vib_power_intensity"].iloc[0] == pytest.approx(expected_rss / 3000.0)


def test_extract_feature_matrix_pipeline(sample_telemetry: pd.DataFrame) -> None:
    config = FeatureConfig(
        include_raw=True,
        rolling_windows=(2,),
        include_transients=True,
        include_domain=True,
    )
    matrix = extract_feature_matrix(sample_telemetry, config)

    # Check timestamp column preserved
    assert matrix.columns[0] == "t"
    assert len(matrix) == len(sample_telemetry)

    # Verify features from each category are present
    assert "V1" in matrix.columns
    assert "V1_roll_mean_2" in matrix.columns
    assert "V1_roc" in matrix.columns
    assert "vib_total_rss" in matrix.columns


def test_extract_feature_matrix_excludes_evaluation_columns() -> None:
    df = pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "V1": [1.0, 2.0, 3.0],
            "V2": [1.0, 2.0, 3.0],
            "V3": [1.0, 2.0, 3.0],
            "V4": [1.0, 2.0, 3.0],
            "V5": [1.0, 2.0, 3.0],
            "V6": [100.0, 200.0, 300.0],
            "fault_state": [0, 0, 1],
            "time_to_next_fault_seconds": [3600.0, 1800.0, 0.0],
            "nearest_fault_distance_seconds": [3600.0, 1800.0, 0.0],
            "source_row": [10, 11, 12],
        }
    )

    matrix = extract_feature_matrix(df)

    # Evaluation columns must NEVER enter predictor feature matrix
    assert "fault_state" not in matrix.columns
    assert "time_to_next_fault_seconds" not in matrix.columns
    assert "nearest_fault_distance_seconds" not in matrix.columns
    assert "source_row" not in matrix.columns


def test_segment_aware_rolling_features_resets_at_outage() -> None:
    # 2 segments separated by an outage:
    # Segment 0: values [10.0, 10.0, 10.0]
    # Segment 1: values [100.0, 100.0, 100.0]
    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 0, 1, 1, 1],
            "V1": [10.0, 10.0, 10.0, 100.0, 100.0, 100.0],
        }
    )

    features = compute_rolling_features(df, windows=(3,), columns=("V1",))

    # At row 3 (first row of segment 1), window=3 must NOT include segment 0's 10.0!
    # Because segment resets, min_periods=1 on segment 1 gives mean = 100.0
    # (If it leaked across the outage, mean would be (10 + 10 + 100)/3 = 40.0)
    assert features["V1_roll_mean_3"].iloc[3] == pytest.approx(100.0)
    assert features["V1_roll_p2p_3"].iloc[3] == pytest.approx(0.0)


def test_segment_aware_transient_features_resets_at_outage() -> None:
    # Segment 0 at 00:00, 00:05
    # Segment 1 after 30-day outage at Day 30 00:00, 00:05
    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 1, 1],
            "t": [
                pd.Timestamp("2026-01-01 00:00:00+0000"),
                pd.Timestamp("2026-01-01 00:05:00+0000"),
                pd.Timestamp("2026-02-01 00:00:00+0000"),
                pd.Timestamp("2026-02-01 00:05:00+0000"),
            ],
            "V1": [10.0, 20.0, 80.0, 90.0],
        }
    )

    transients = compute_transient_features(df, columns=("V1",))

    # First row of segment 1 (index 2) must reset derivative to 0.0 rather than
    # calculating (80 - 20) / 30 days!
    assert transients["V1_roc"].iloc[2] == pytest.approx(0.0)
    assert transients["V1_acc"].iloc[2] == pytest.approx(0.0)

    # Second row of segment 1 (index 3) is within segment: (90 - 80) / 300s
    assert transients["V1_roc"].iloc[3] == pytest.approx(10.0 / 300.0)


def test_domain_features_min_power_floor() -> None:
    df = pd.DataFrame(
        {
            "V1": [1.0],
            "V2": [1.0],
            "V3": [1.0],
            "V4": [1.0],
            "V5": [20.0],
            "V6": [5.0],  # Very low power (< min_power_floor 50.0)
        }
    )

    domain_default = compute_domain_features(df, min_power_floor=50.0)
    # total_rss = sqrt(4) = 2.0
    # Clamped power = 50.0 -> vib_power_intensity = 2.0 / 50.0 = 0.04
    assert domain_default["vib_power_intensity"].iloc[0] == pytest.approx(2.0 / 50.0)

    # Custom power floor
    domain_custom = compute_domain_features(df, min_power_floor=10.0)
    assert domain_custom["vib_power_intensity"].iloc[0] == pytest.approx(2.0 / 10.0)
