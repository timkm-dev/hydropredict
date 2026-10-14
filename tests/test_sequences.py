from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydropredict.data.sequences import build_sequences


def observations(count: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=count, freq="5min", tz="UTC"),
            **{f"V{i}": np.arange(count, dtype=float) for i in range(1, 7)},
        }
    )


def test_sequence_alignment_and_future_independence() -> None:
    data = observations()
    complete = build_sequences(data.assign(fault_state=2, time_to_next_fault_seconds=0))
    prefix = build_sequences(data.iloc[:20])
    np.testing.assert_array_equal(prefix.values, complete.values[: len(prefix.values)])
    assert complete.values.shape == (24, 7, 6)
    assert complete.endpoints.t.iloc[0] - complete.endpoints.window_start.iloc[0] == pd.Timedelta(
        minutes=30
    )
    assert complete.values[0, -1, 0] == 6


def test_sequences_reset_at_gaps_and_partition_boundaries() -> None:
    data = observations()
    data.loc[15:, "t"] += pd.Timedelta(days=1)
    result = build_sequences(data)
    assert len(result.values) == 18
    assert result.accounting["segments"] == 2
    assert result.accounting["warmup_excluded_observations"] == 12
    assert (result.endpoints.t - result.endpoints.window_start).max() == pd.Timedelta(minutes=30)
    assert len(build_sequences(data.iloc[15:20]).values) == 0


def test_invalid_cadence_and_values_are_handled() -> None:
    data = observations()
    data.loc[10, "t"] = data.t.iloc[9] + pd.Timedelta(seconds=60)
    assert build_sequences(data).accounting["cadence_resets"] == 2
    with pytest.raises(ValueError):
        build_sequences(data.iloc[::-1])
    data.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_sequences(data)
