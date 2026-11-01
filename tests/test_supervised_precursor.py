"""Leakage and boundary regressions for the corrected supervised experiment."""

import numpy as np
import pandas as pd
import pytest

from hydropredict.data.context import reconstruct_context
from scripts.supervised_precursor import features, groups, labels, persistent, split_rows


def telemetry(times):
    return pd.DataFrame({"t": times, **{f"V{i}": np.arange(len(times)) + i for i in range(1, 7)}})


def test_causal_features_prefix_and_gap_reset():
    times = pd.date_range("2018-12-01", periods=30, freq="5min", tz="UTC")
    frame = telemetry(times)
    full = features(frame)
    prefix = features(frame.iloc[:15])
    pd.testing.assert_frame_equal(prefix, full[full.t.le(times[14])].reset_index(drop=True))
    changed = frame.copy()
    changed.loc[15:, "V1"] = 100000
    pd.testing.assert_frame_equal(
        prefix, features(changed).query("t <= @times[14]").reset_index(drop=True)
    )
    broken = frame.copy()
    broken.loc[15:, "t"] += pd.Timedelta(hours=2)
    assert not features(broken).t.between(broken.t.iloc[15], broken.t.iloc[20]).any()
    short = frame.copy()
    short.loc[15:, "t"] -= pd.Timedelta(minutes=4)
    assert not features(short).t.between(short.t.iloc[15], short.t.iloc[20]).any()


def test_targets_and_boundary_group_purge():
    faults = pd.to_datetime(["2018-12-31 23:00", "2019-01-01 02:00", "2019-01-02 00:00"], utc=True)
    assert [len(g) for g in groups(faults)] == [2, 1]
    times = pd.to_datetime(["2018-12-31 17:00", "2018-12-31 16:59", "2019-01-02 00:00"], utc=True)
    assert labels(times, faults).tolist() == [1, 0, 0]
    frame = telemetry(pd.date_range("2018-12-31", "2019-01-02 00:00", freq="5min", tz="UTC"))
    fit = split_rows(
        frame, faults, pd.Timestamp("2018-12-01", tz="UTC"), pd.Timestamp("2019-01-01", tz="UTC")
    )
    cal = split_rows(
        frame, faults, pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-02-01", tz="UTC")
    )
    assert fit.t.max() < pd.Timestamp("2018-12-31 17:00", tz="UTC")
    assert cal.t.min() > pd.Timestamp("2019-01-01 02:00", tz="UTC")
    assert not set(fit.t) & set(cal.t)


def test_union_duplicate_conflict_and_provenance():
    frame = telemetry(pd.date_range("2018-12-01", periods=4, freq="5min", tz="UTC"))
    start, end = frame.t.min(), frame.t.max() + pd.Timedelta(seconds=1)
    union, provenance, report = reconstruct_context({"a": frame, "b": frame}, start, end)
    assert len(union) == 4 and report["identical_duplicates_collapsed"] == 4
    assert "source_records" not in union and "source_records" in provenance
    conflict = frame.copy()
    conflict.loc[0, "V1"] += 1
    with pytest.raises(ValueError, match="Conflicting"):
        reconstruct_context({"a": frame, "b": conflict}, start, end)


def test_persistence_resets_on_missing_or_too_close_readings():
    times = pd.Series(
        pd.to_datetime(
            [
                "2019-01-01 00:00",
                "2019-01-01 00:05",
                "2019-01-01 00:10",
                "2019-01-01 00:30",
                "2019-01-01 00:31",
                "2019-01-01 00:36",
            ],
            utc=True,
        )
    )
    assert persistent(times, pd.Series([True] * 6)).tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
