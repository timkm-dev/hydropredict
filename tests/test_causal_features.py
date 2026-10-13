from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydropredict.features.causal import causal_feature_frame
from hydropredict.models.isolation_forest import SHPIsolationForest


def readings(count: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(count, 6)), columns=[f"V{i}" for i in range(1, 7)])
    frame.insert(0, "t", pd.date_range("2026-01-01", periods=count, freq="5min", tz="UTC"))
    return frame


def test_temporal_features_are_prefix_invariant_and_ignore_outcomes() -> None:
    data = readings()
    expected = causal_feature_frame(data.iloc[:100], "temporal")
    complete = causal_feature_frame(data.assign(fault_state=2), "temporal")
    pd.testing.assert_frame_equal(expected, complete.iloc[: len(expected)].reset_index(drop=True))
    assert expected.t.iloc[0] == data.t.iloc[24]
    assert len(expected.columns) == 37
    assert "fault_state" not in expected


def test_temporal_windows_reset_after_outage_and_use_actual_time() -> None:
    data = readings(60)
    data.loc[30:, "t"] += pd.Timedelta(days=1)
    result = causal_feature_frame(data, "temporal")
    assert result.t.tolist() == data.t.iloc[list(range(24, 30)) + list(range(54, 60))].tolist()
    with pytest.raises(ValueError):
        causal_feature_frame(data.iloc[::-1], "temporal")


def test_temporal_model_roundtrip_and_warmup(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    data = readings()
    model = SHPIsolationForest(n_estimators=20, feature_mode="temporal")
    model.fit(data.iloc[:400], data.iloc[400:], version="temporal-test")
    expected = model.score(data.iloc[400:])
    assert len(expected) == 176
    model.save(tmp_path / "bundle", {})
    actual = SHPIsolationForest.load(tmp_path / "bundle").score(data.iloc[400:])
    pd.testing.assert_frame_equal(expected, actual)


def test_derivative_uses_real_elapsed_seconds() -> None:
    data = readings(40)
    data["V1"] = np.arange(40, dtype=float)
    data.loc[39, "t"] += pd.Timedelta(seconds=30)
    result = causal_feature_frame(data, "temporal")
    assert result.V1_roc.iloc[-1] == pytest.approx(1 / 330)
