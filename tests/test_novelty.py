"""Normal-only calibration, frozen round trips and causal scoring contracts."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from hydropredict.analysis.novelty import CANDIDATES, NormalNovelty


def data() -> pd.DataFrame:
    rng = np.random.default_rng(24)
    frame = pd.DataFrame(rng.normal(size=(650, 6)), columns=[f"V{i}" for i in range(1, 7)])
    frame.insert(0, "t", pd.date_range("2020-01-01", periods=len(frame), freq="5min", tz="UTC"))
    return frame


@pytest.mark.parametrize("method", CANDIDATES)
def test_calibration_no_future_leakage_and_round_trip(method: str) -> None:
    frame = data()
    fitting, calibration, evaluation = frame.iloc[:350], frame.iloc[350:500], frame.iloc[500:]
    model = NormalNovelty(method).fit(fitting, calibration)
    columns = model.features(fitting)
    assert model.scaler.means["V1"] == pytest.approx(columns.V1.mean())
    assert model.threshold == pytest.approx(
        np.quantile(model.score(calibration).anomaly_score, 0.99)
    )
    expected = model.score(evaluation)
    # Labels/source membership must never be features.
    actual = model.score(evaluation.assign(fault_state=2, source="fault"))
    pd.testing.assert_frame_equal(expected, actual)
    restored = pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(expected, restored.score(evaluation))
    prefix = model.score(evaluation.iloc[:80])
    np.testing.assert_allclose(
        prefix.anomaly_score, expected[expected.t.isin(prefix.t)].anomaly_score, rtol=1e-10
    )


def test_temporal_gap_and_unavailable_rows_are_excluded() -> None:
    frame = data()
    model = NormalNovelty("local_temporal").fit(frame.iloc[:350], frame.iloc[350:500])
    evaluation = frame.iloc[500:].copy()
    evaluation.loc[evaluation.index[50:], "t"] += pd.Timedelta(hours=3)
    scores = model.score(evaluation)
    assert len(scores) == len(evaluation) - 12
    assert model.score(evaluation.iloc[:3]).empty
    assert frame.iloc[500].t not in set(scores.t)


def test_rejects_partition_overlap_and_nonfinite_input() -> None:
    frame = data()
    with pytest.raises(ValueError, match="precede"):
        NormalNovelty("local_level").fit(frame.iloc[:350], frame.iloc[300:500])
    frame.loc[0, "V1"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        NormalNovelty("local_level").fit(frame.iloc[:350], frame.iloc[350:500])
