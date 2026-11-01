"""Verify retrospective association never counts post-fault alerts as warnings."""

import pandas as pd

from scripts.map_anomalies_to_faults import episodes, map_events


def test_fault_boundary_and_missing_scores():
    fault = pd.Timestamp("2019-02-03", tz="UTC")
    frame = pd.DataFrame(
        {
            "t": pd.date_range(fault - pd.Timedelta(hours=6), periods=73, freq="5min"),
            "anomaly_score": [0.0] * 72 + [2.0],
            "threshold": [1.0] * 73,
            "is_alert": [False] * 72 + [True],
        }
    )
    result = map_events(frame, [fault])[0]
    assert result["status"] == "missed"
    assert result["alert_points"] == 0
    frame.loc[0, ["is_alert", "anomaly_score"]] = [True, 2.0]
    result = map_events(frame, [fault])[0]
    assert result["status"] == "warned"
    assert result["lead_minutes"] == 360
    result = map_events(frame.iloc[-2:], [fault])[0]
    assert result["status"] == "insufficient"


def test_episodes_reset_at_normal_points_gaps_and_segments():
    frame = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2019-02-01 00:00Z",
                    "2019-02-01 00:05Z",
                    "2019-02-01 00:10Z",
                    "2019-02-01 00:15Z",
                    "2019-02-01 02:00Z",
                    "2019-02-01 02:05Z",
                ]
            ),
            "is_alert": [True, True, False, True, True, True],
            "segment_id": [0, 0, 0, 0, 1, 2],
        }
    )
    assert [e["points"] for e in episodes(frame)] == [2, 1, 1, 1]


def test_overlapping_windows_share_a_group():
    faults = list(pd.to_datetime(["2019-02-01 06:00Z", "2019-02-01 07:00Z"]))
    frame = pd.DataFrame(
        {
            "t": pd.to_datetime(["2019-02-01 05:00Z"]),
            "is_alert": [True],
            "anomaly_score": [2.0],
            "threshold": [1.0],
        }
    )
    events = map_events(frame, faults)
    assert events[0]["group"] == events[1]["group"]
    assert all(e["alert_points"] == 1 for e in events)
