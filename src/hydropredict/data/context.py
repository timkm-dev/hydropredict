"""Reconstruct observed context within one development interval without filling gaps."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS, REQUIRED_COLUMNS


def reconstruct_context(
    sources: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Merge exact source timestamps; reject conflicting duplicates, never synthesise rows.

    Provenance is returned separately and must not be passed as predictor features.
    Boundaries are inclusive start, exclusive end. No historical context crosses them.
    """
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Context boundaries must be valid and ordered.")
    parts = []
    counts = {}
    for name, frame in sources.items():
        work = frame[list(REQUIRED_COLUMNS)].copy()
        work["t"] = pd.to_datetime(work.t, utc=True)
        if work.t.isna().any():
            raise ValueError("Invalid source timestamps.")
        work["source_row"] = frame.get("source_row", pd.Series(frame.index + 2, index=frame.index))
        work = work[(work.t >= start) & (work.t < end)].copy()
        if not np.isfinite(work[list(MEASUREMENT_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError("Context measurements must be finite.")
        counts[name] = len(work)
        work["source_id"] = name
        parts.append(work)
    if not parts or sum(counts.values()) == 0:
        raise ValueError("No source observations within context boundaries.")
    combined = pd.concat(parts, ignore_index=True).sort_values("t", kind="stable")
    duplicate_times = combined.loc[combined.t.duplicated(keep=False), "t"]
    for timestamp in duplicate_times.unique():
        rows = combined[combined.t == timestamp]
        if len(rows[list(MEASUREMENT_COLUMNS)].drop_duplicates()) != 1:
            raise ValueError(f"Conflicting source observations at {timestamp}")
    provenance_rows = []
    for timestamp, group in combined.groupby("t", sort=True):
        records = [
            {"source_id": str(row.source_id), "source_row": int(str(row.source_row))}
            for row in group.itertuples()
        ]
        provenance_rows.append({"t": timestamp, "source_records": json.dumps(records)})
    provenance = pd.DataFrame(provenance_rows)
    timeline = combined.drop_duplicates("t")[list(REQUIRED_COLUMNS)].reset_index(drop=True)
    gaps = timeline.t.diff().dt.total_seconds()
    report = {
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "source_rows": counts,
        "input_rows": sum(counts.values()),
        "timeline_rows": len(timeline),
        "identical_duplicates_collapsed": len(combined) - len(timeline),
        "outages_over_450s": int(gaps.gt(450).sum()),
        "synthetic_rows": 0,
        "metadata_is_predictor": False,
    }
    return timeline, provenance, report
