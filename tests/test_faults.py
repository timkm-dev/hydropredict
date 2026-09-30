from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from hydropredict.data.faults import (
    FaultAlignmentConfig,
    FaultState,
    align_fault_labels,
    align_telemetry_file,
    load_fault_log,
)


def write_csv(path: Path, rows: list[list[str]], header: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_load_fault_log(tmp_path: Path) -> None:
    faults_csv = tmp_path / "faults.csv"
    write_csv(
        faults_csv,
        [["2026-01-01 12:00:00"], ["2026-01-01 06:00:00"]],
        header=["t"],
    )

    loaded = load_fault_log(faults_csv)
    assert len(loaded) == 2
    # Verify sorted chronologically
    assert loaded[0] == pd.Timestamp("2026-01-01 06:00:00+0000")
    assert loaded[1] == pd.Timestamp("2026-01-01 12:00:00+0000")


def test_load_fault_log_empty_and_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fault_log(tmp_path / "missing.csv")

    empty_csv = tmp_path / "empty.csv"
    write_csv(empty_csv, [], header=["t"])
    assert load_fault_log(empty_csv) == []


def test_align_fault_labels_states() -> None:
    # Telemetry hourly from 00:00 to 05:00
    times = pd.date_range("2026-01-01 00:00:00", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"t": times, "V1": [1.0] * 6})

    # Fault at 03:00
    # Config: 2 hours precursor (01:00 to 03:00 exclusive), 30 min fault duration
    fault = pd.Timestamp("2026-01-01 03:00:00+0000")
    cfg = FaultAlignmentConfig(precursor_window_seconds=7200.0, fault_duration_seconds=1800.0)

    labeled = align_fault_labels(df, [fault], config=cfg)

    # 00:00 -> NORMAL (more than 2h before 03:00)
    # 01:00 -> PRECURSOR (2h before)
    # 02:00 -> PRECURSOR (1h before)
    # 03:00 -> FAULT (active at fault time)
    # 04:00 -> NORMAL (past fault duration)
    # 05:00 -> NORMAL
    states = labeled["fault_state"].tolist()
    assert states[0] == FaultState.NORMAL.value
    assert states[1] == FaultState.PRECURSOR.value
    assert states[2] == FaultState.PRECURSOR.value
    assert states[3] == FaultState.FAULT.value
    assert states[4] == FaultState.NORMAL.value
    assert states[5] == FaultState.NORMAL.value

    # Countdown to next fault
    # At 01:00, fault is at 03:00 -> 7200 seconds
    assert labeled["time_to_next_fault_seconds"].iloc[1] == pytest.approx(7200.0)
    # After fault at 04:00, no next fault -> NaN
    assert pd.isna(labeled["time_to_next_fault_seconds"].iloc[4])


def test_align_fault_labels_empty_faults() -> None:
    df = pd.DataFrame({"t": [pd.Timestamp("2026-01-01 00:00:00+0000")], "V1": [1.0]})
    labeled = align_fault_labels(df, [])
    assert labeled["fault_state"].iloc[0] == FaultState.NORMAL.value
    assert pd.isna(labeled["time_to_next_fault_seconds"].iloc[0])


def test_align_telemetry_file_end_to_end(tmp_path: Path) -> None:
    tel_csv = tmp_path / "tel.csv"
    fault_csv = tmp_path / "fault.csv"
    out_csv = tmp_path / "aligned.csv"

    write_csv(
        tel_csv,
        [["2026-01-01 00:00:00", "1.0"], ["2026-01-01 01:00:00", "1.2"]],
        header=["t", "V1"],
    )
    write_csv(fault_csv, [["2026-01-01 01:00:00"]], header=["t"])

    df = align_telemetry_file(tel_csv, fault_csv, output_csv=out_csv)
    assert out_csv.exists()
    assert "fault_state" in df.columns
    assert len(df) == 2
