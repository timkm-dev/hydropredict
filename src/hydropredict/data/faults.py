"""Ground-truth fault event loading, alignment, and evaluation labeling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from hydropredict.data.schema import TIMESTAMP_COLUMN


class FaultState(IntEnum):
    """Categorical operational health states."""

    NORMAL = 0
    PRECURSOR = 1
    FAULT = 2


@dataclass(frozen=True, slots=True)
class FaultAlignmentConfig:
    """Temporal window parameters for predictive maintenance labeling.

    Attributes:
        precursor_window_seconds: Warning window preceding a fault event where
            incipient degradation occurs (default 6 hours = 21,600 seconds).
        fault_duration_seconds: Duration attributed to the active fault event and
            immediate shutdown (default 30 minutes = 1,800 seconds).
    """

    precursor_window_seconds: float = 21600.0
    fault_duration_seconds: float = 1800.0


def load_fault_log(source: Path | str) -> list[pd.Timestamp]:
    """Load ground-truth fault timestamps from CSV.

    Expects a CSV with a timestamp column ('t' or single column),
    returns a sorted list of UTC pd.Timestamps.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Fault log not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return []

    # Use first column if 't' not explicitly named
    col = TIMESTAMP_COLUMN if TIMESTAMP_COLUMN in df.columns else df.columns[0]
    timestamps = pd.to_datetime(df[col], utc=True).dropna().sort_values().tolist()
    return timestamps


def align_fault_labels(
    telemetry_df: pd.DataFrame,
    fault_timestamps: Sequence[pd.Timestamp],
    config: FaultAlignmentConfig | None = None,
) -> pd.DataFrame:
    """Label telemetry records against ground-truth failure logs.

    Computes:
      - fault_state:
          0 (NORMAL): Healthy operation outside anomaly lead-up windows.
          1 (PRECURSOR): Incipient fault warning period preceding a failure.
          2 (FAULT): Active failure / trip event.
      - time_to_next_fault_seconds: Remaining time until next recorded fault event.
      - nearest_fault_distance_seconds: Absolute distance to closest fault.
    """
    cfg = config or FaultAlignmentConfig()
    result = telemetry_df.copy()

    if TIMESTAMP_COLUMN not in result.columns:
        raise ValueError(f"DataFrame must contain '{TIMESTAMP_COLUMN}' column.")

    t_series = pd.to_datetime(result[TIMESTAMP_COLUMN], utc=True)
    n_rows = len(result)

    fault_state = np.full(n_rows, FaultState.NORMAL.value, dtype=np.int32)
    time_to_next = np.full(n_rows, np.nan, dtype=np.float64)
    nearest_dist = np.full(n_rows, np.nan, dtype=np.float64)

    if not fault_timestamps:
        result["fault_state"] = fault_state
        result["time_to_next_fault_seconds"] = time_to_next
        result["nearest_fault_distance_seconds"] = nearest_dist
        return result

    fault_times_ns = np.array([ts.value for ts in fault_timestamps], dtype=np.int64)
    t_vals_ns = t_series.values.astype("datetime64[ns]").astype(np.int64)

    precursor_ns = int(cfg.precursor_window_seconds * 1e9)
    fault_dur_ns = int(cfg.fault_duration_seconds * 1e9)

    for i, t_val in enumerate(t_vals_ns):
        # Time differences: positive if fault is in future, negative if in past
        diff_ns = fault_times_ns - t_val

        # Distance in seconds to closest fault
        abs_diff_s = np.abs(diff_ns) / 1e9
        nearest_dist[i] = float(np.min(abs_diff_s))

        # Next upcoming fault
        future_diffs = diff_ns[diff_ns >= 0]
        if len(future_diffs) > 0:
            time_to_next[i] = float(np.min(future_diffs) / 1e9)

        # Check if in active fault window: [0, fault_dur_ns]
        is_active = np.any((diff_ns <= 0) & (diff_ns >= -fault_dur_ns))
        if is_active:
            fault_state[i] = FaultState.FAULT.value
            continue

        # Check if in precursor window: [0, precursor_ns]
        is_precursor = np.any((diff_ns > 0) & (diff_ns <= precursor_ns))
        if is_precursor:
            fault_state[i] = FaultState.PRECURSOR.value

    result["fault_state"] = fault_state
    result["time_to_next_fault_seconds"] = time_to_next
    result["nearest_fault_distance_seconds"] = nearest_dist

    return result


def align_telemetry_file(
    telemetry_csv: Path | str,
    faults_csv: Path | str,
    output_csv: Path | str | None = None,
    config: FaultAlignmentConfig | None = None,
) -> pd.DataFrame:
    """End-to-end alignment from files, optionally persisting to output CSV."""
    telemetry_path = Path(telemetry_csv)
    faults_path = Path(faults_csv)

    df = pd.read_csv(telemetry_path)
    faults = load_fault_log(faults_path)
    aligned = align_fault_labels(df, faults, config=config)

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aligned.to_csv(out_path, index=False)

    return aligned
