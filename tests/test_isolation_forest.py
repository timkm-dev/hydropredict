from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydropredict.models.isolation_forest import SHPIsolationForest


def reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(500, 6)), columns=[f"V{i}" for i in range(1, 7)])
    data.insert(0, "t", pd.date_range("2026-01-01", periods=500, freq="5min", tz="UTC"))
    return data.iloc[:350].copy(), data.iloc[350:].copy()


def test_model_roundtrip_and_outcome_columns_cannot_affect_scores(tmp_path: Path) -> None:
    fitting, calibration = reference_data()
    model = SHPIsolationForest(n_estimators=20).fit(fitting, calibration, version="test-v1")
    expected = model.score(calibration)
    enriched = calibration.assign(fault_state=2, time_to_next_fault_seconds=0, segment_id=999)
    np.testing.assert_array_equal(expected.anomaly_score, model.score(enriched).anomaly_score)
    bundle = tmp_path / "bundle"
    model.save(bundle, {"purpose": "test"})
    actual = SHPIsolationForest.load(bundle).score(calibration)
    pd.testing.assert_frame_equal(expected, actual)
    assert model.scaler.means["V1"] == pytest.approx(fitting.V1.mean())
    with pytest.raises(FileExistsError):
        model.save(bundle, {})
    (bundle / "scaler.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SHPIsolationForest.load(bundle)


def test_predictions_are_pointwise_and_threshold_equality_is_not_alert() -> None:
    fitting, calibration = reference_data()
    model = SHPIsolationForest(n_estimators=20).fit(fitting, calibration, version="test-v1")
    baseline = model.score(calibration)
    changed = calibration.copy()
    changed.loc[changed.index[-1], "V1"] = 1e6
    np.testing.assert_array_equal(
        baseline.anomaly_score.iloc[:-1], model.score(changed).anomaly_score.iloc[:-1]
    )
    model.threshold = float(baseline.anomaly_score.iloc[0])
    assert not model.score(calibration.iloc[:1]).is_alert.iloc[0]


def test_model_rejects_overlap_invalid_values_and_unfitted_scoring() -> None:
    fitting, calibration = reference_data()
    with pytest.raises(RuntimeError):
        SHPIsolationForest().score(calibration)
    with pytest.raises(ValueError, match="precede"):
        SHPIsolationForest().fit(fitting, fitting.iloc[-100:], version="invalid")
    fitting.loc[fitting.index[0], "V1"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        SHPIsolationForest().fit(fitting, calibration, version="invalid")
