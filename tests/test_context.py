from __future__ import annotations

import pandas as pd
import pytest

from hydropredict.data.context import reconstruct_context
from hydropredict.features.causal import causal_feature_frame


def samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC"),
            **{f"V{i}": [float(j) for j in range(60)] for i in range(1, 7)},
        }
    )


def test_context_restores_artificial_split_without_provenance_features() -> None:
    all_rows = samples()
    timeline, provenance, report = reconstruct_context(
        {"normal": all_rows.iloc[:40], "prefault": all_rows.iloc[40:]},
        all_rows.t.iloc[0],
        all_rows.t.iloc[-1] + pd.Timedelta(minutes=5),
    )
    pd.testing.assert_frame_equal(timeline, all_rows)
    assert report["synthetic_rows"] == 0
    assert len(causal_feature_frame(timeline, "temporal")) == 36
    assert len(causal_feature_frame(all_rows.iloc[40:], "temporal")) == 0
    assert "source_records" not in timeline
    assert len(provenance) == 60


def test_context_preserves_outages_and_interval_boundaries() -> None:
    data = samples()
    data.loc[30:, "t"] += pd.Timedelta(days=2)
    timeline, _, report = reconstruct_context(
        {"a": data},
        data.t.iloc[1],
        data.t.iloc[-1],
    )
    assert len(timeline) == 58
    assert report["outages_over_450s"] == 1
    assert len(causal_feature_frame(timeline, "temporal")) == 10


def test_exact_duplicates_have_provenance_and_conflicts_fail() -> None:
    data = samples().iloc[:1].copy()
    start, end = data.t.iloc[0], data.t.iloc[0] + pd.Timedelta(hours=1)
    timeline, provenance, report = reconstruct_context({"a": data, "b": data}, start, end)
    assert len(timeline) == 1
    assert report["identical_duplicates_collapsed"] == 1
    assert '"a"' in provenance.source_records.iloc[0]
    assert '"b"' in provenance.source_records.iloc[0]
    changed = data.copy()
    changed["V1"] = 99.0
    with pytest.raises(ValueError, match="Conflicting"):
        reconstruct_context({"a": data, "b": changed}, start, end)


def test_context_future_values_cannot_change_earlier_features() -> None:
    data = samples()
    start, end = data.t.iloc[0], data.t.iloc[-1] + pd.Timedelta(minutes=5)
    expected, _, _ = reconstruct_context({"a": data.iloc[:40]}, start, end)
    complete, _, _ = reconstruct_context({"a": data.iloc[:40], "b": data.iloc[40:]}, start, end)
    prefix = causal_feature_frame(expected, "temporal")
    full = causal_feature_frame(complete, "temporal")
    pd.testing.assert_frame_equal(prefix, full.iloc[: len(prefix)].reset_index(drop=True))
