"""Causal six-hour trend features for the fixed normal-only study."""

import numpy as np
import pandas as pd


def trend_features(frame: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Six elapsed hours, no interpolation, reset after a gap over 450 seconds."""
    frame = frame.reset_index(drop=True).copy()
    frame["t"] = pd.to_datetime(frame["t"], utc=True)
    if frame.t.isna().any() or frame.t.duplicated().any() or not frame.t.is_monotonic_increasing:
        raise ValueError("Invalid or duplicate timeline timestamps")
    if not np.isfinite(frame[channels].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite telemetry")
    segments = frame.t.diff().dt.total_seconds().gt(450).cumsum()
    output = []
    for _, group in frame.groupby(segments):
        times = pd.DatetimeIndex(group.t)
        hours = (times - times[0]).total_seconds().to_numpy() / 3600
        values = group[channels].to_numpy(dtype=float)
        for end in range(len(group)):
            if hours[end] < 6:
                continue
            begin = times.searchsorted(times[end] - pd.Timedelta(hours=6))
            x, y = hours[begin : end + 1], values[begin : end + 1]
            # Coverage in distinct five-minute slots, not raw row counts.
            slots = np.floor((hours[end] - x) * 12 + 1e-8).astype(int)
            if len(np.unique(slots)) < 58:
                continue
            centered = x - x.mean()
            slope = centered @ (y - y.mean(axis=0)) / (centered @ centered)
            row: dict[str, object] = {"t": times[end]}
            for j, channel in enumerate(channels):
                row[f"{channel}_slope_per_hour"] = slope[j]
                row[f"{channel}_std_6h"] = y[:, j].std(ddof=1)
            output.append(row)
    columns = ["t"] + [
        name for channel in channels for name in (f"{channel}_slope_per_hour", f"{channel}_std_6h")
    ]
    return pd.DataFrame(output, columns=columns)
