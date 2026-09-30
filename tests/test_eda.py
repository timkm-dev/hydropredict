from __future__ import annotations

import pandas as pd
import pytest

from hydropredict.analysis.eda import (
    compute_health_distribution_profile,
    format_eda_summary,
)


def test_compute_health_distribution_profile() -> None:
    df = pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC"),
            "fault_state": [0, 0, 0, 1, 1, 2],
            "V1": [1.0, 1.0, 1.0, 2.0, 2.0, 0.0],  # normal mean 1.0, precursor mean 2.0 (+100%)
            "V2": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )

    profile = compute_health_distribution_profile(df)
    assert profile["total_records"] == 6
    assert profile["normal_records"] == 3
    assert profile["precursor_records"] == 2
    assert profile["fault_records"] == 1

    v1_metrics = profile["metrics"]["V1"]
    assert v1_metrics["normal_mean"] == pytest.approx(1.0)
    assert v1_metrics["precursor_mean"] == pytest.approx(2.0)
    assert v1_metrics["percentage_shift"] == pytest.approx(100.0)

    summary_text = format_eda_summary(profile)
    assert "HydroPredict - Telemetry Exploratory Distribution Analysis" in summary_text
    assert "V1" in summary_text


def test_eda_missing_fault_state_raises() -> None:
    df = pd.DataFrame({"V1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="fault_state"):
        compute_health_distribution_profile(df)
