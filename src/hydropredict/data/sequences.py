"""Causal, observed-only sequences with explicit gap and warm-up accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hydropredict.data.schema import MEASUREMENT_COLUMNS


@dataclass(frozen=True)
class SequenceBatch:
    values: np.ndarray[Any, np.dtype[np.float32]]
    endpoints: pd.DataFrame
    accounting: dict[str, int]


def build_sequences(
    frame: pd.DataFrame,
    sequence_length: int = 7,
    minimum_step_seconds: float = 150,
    maximum_step_seconds: float = 450,
) -> SequenceBatch:
    """Seven nominal five-minute observations span thirty minutes, not thirty-five.

    A sequence ends at its last reading. No labels, filling or future observations
    enter the tensor. Each call starts a new partition and resets context.
    """
    if sequence_length < 2 or not 0 < minimum_step_seconds <= maximum_step_seconds:
        raise ValueError("Invalid sequence length or cadence limits.")
    timestamps = pd.to_datetime(frame["t"], utc=True)
    if (
        timestamps.isna().any()
        or timestamps.duplicated().any()
        or not timestamps.is_monotonic_increasing
    ):
        raise ValueError("Sequence timestamps must be valid, unique and chronological.")
    measurements = frame[list(MEASUREMENT_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(measurements).all():
        raise ValueError("Sequence measurements must be finite.")
    dt = timestamps.diff().dt.total_seconds()
    resets = dt.isna() | (dt < minimum_step_seconds) | (dt > maximum_step_seconds)
    boundaries = np.flatnonzero(resets.to_numpy()).tolist() + [len(frame)]
    sequences = []
    ends = []
    starts = []
    segments = []
    for segment, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        for end in range(start + sequence_length - 1, stop):
            sequences.append(measurements[end - sequence_length + 1 : end + 1])
            ends.append(end)
            starts.append(end - sequence_length + 1)
            segments.append(segment)
    values = (
        np.stack(sequences)
        if sequences
        else np.empty((0, sequence_length, len(MEASUREMENT_COLUMNS)), dtype=np.float32)
    )
    endpoints = pd.DataFrame(
        {
            "t": timestamps.iloc[np.asarray(ends, dtype=int)].to_numpy(),
            "window_start": timestamps.iloc[np.asarray(starts, dtype=int)].to_numpy(),
            "input_end_position": ends,
            "segment_id": segments,
        }
    )
    return SequenceBatch(
        values,
        endpoints,
        {
            "input_observations": len(frame),
            "eligible_sequences": len(ends),
            "warmup_excluded_observations": len(frame) - len(ends),
            "segments": max(0, len(boundaries) - 1),
            "cadence_resets": max(0, int(resets.sum()) - 1),
            "synthetic_observations": 0,
        },
    )
