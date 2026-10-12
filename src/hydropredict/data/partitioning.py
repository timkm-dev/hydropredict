"""Dataset partitioning and chronological non-leakage contracts for HydroPredict."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from hydropredict.data.faults import load_fault_log
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.data.schema import REQUIRED_COLUMNS, TIMESTAMP_COLUMN


def compute_file_sha256(path: Path | str) -> str:
    """Compute uppercase hex SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


@dataclass(frozen=True, slots=True)
class PartitionMetadata:
    """Metadata describing one dataset partition."""

    partition_name: str
    relative_path: str
    record_count: int
    start_time: str
    end_time: str
    sha256: str
    source_origins: tuple[str, ...]
    fault_events_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_name": self.partition_name,
            "relative_path": self.relative_path,
            "record_count": self.record_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sha256": self.sha256,
            "source_origins": list(self.source_origins),
            "fault_events_count": self.fault_events_count,
        }


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """Complete provenance and boundary manifest for a partitioned dataset."""

    strategy: str
    created_at: str
    is_strictly_chronological: bool
    partitions: dict[str, PartitionMetadata]
    scope_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "created_at": self.created_at,
            "is_strictly_chronological": self.is_strictly_chronological,
            "partitions": {name: meta.to_dict() for name, meta in self.partitions.items()},
            "scope_notes": self.scope_notes,
        }

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> PartitionManifest:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        parts: dict[str, PartitionMetadata] = {}
        for name, p_data in data.get("partitions", {}).items():
            parts[name] = PartitionMetadata(
                partition_name=p_data["partition_name"],
                relative_path=p_data["relative_path"],
                record_count=int(p_data["record_count"]),
                start_time=p_data["start_time"],
                end_time=p_data["end_time"],
                sha256=p_data["sha256"],
                source_origins=tuple(p_data.get("source_origins", [])),
                fault_events_count=int(p_data.get("fault_events_count", 0)),
            )

        return cls(
            strategy=data["strategy"],
            created_at=data["created_at"],
            is_strictly_chronological=bool(data["is_strictly_chronological"]),
            partitions=parts,
            scope_notes=data.get("scope_notes", ""),
        )


def create_shp_partitions(
    train_source: Path | str = "data/raw/shp_train.csv",
    test_source: Path | str = "data/raw/shp_test.csv",
    faults_source: Path | str = "data/raw/shp_faults.csv",
    output_dir: Path | str = "data/processed/partitions",
    strategy: Literal["chronological", "published"] = "chronological",
    train_end_boundary: str = "2019-02-01T00:00:00Z",
    val_end_boundary: str = "2019-05-01T00:00:00Z",
) -> PartitionManifest:
    """Create auditable, traceable partitions for Small Hydroelectric Plant telemetry.

    Supports two strategies:
    1. 'chronological': Strictly separates train (< train_end_boundary),
       validation (train_end to val_end), and test (>= val_end).
       Ensures max(train) < min(val) < min(test) with zero temporal leakage.
    2. 'published': Preserves the exact de Santis & Costa (2020) source split
       (47,857 train, 4,897 test) as quoted in the project report.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_train = load_telemetry_dataframe(train_source)
    raw_test = load_telemetry_dataframe(test_source)
    fault_timestamps = load_fault_log(faults_source) if Path(faults_source).exists() else []

    parts_dict: dict[str, PartitionMetadata] = {}
    is_chronological = False

    if strategy == "chronological":
        is_chronological = True
        t_split_train = pd.Timestamp(train_end_boundary)
        t_split_val = pd.Timestamp(val_end_boundary)

        # Train partition from normal operating source up to cutoff
        df_train = raw_train[raw_train[TIMESTAMP_COLUMN] < t_split_train].copy()
        # Validation partition from test source between cutoffs
        df_val = raw_test[
            (raw_test[TIMESTAMP_COLUMN] >= t_split_train)
            & (raw_test[TIMESTAMP_COLUMN] < t_split_val)
        ].copy()
        # Final test partition from test source after val cutoff
        df_test = raw_test[raw_test[TIMESTAMP_COLUMN] >= t_split_val].copy()

        # Verify strict non-overlap and temporal ordering
        if not df_train.empty and not df_val.empty:
            assert df_train[TIMESTAMP_COLUMN].max() < df_val[TIMESTAMP_COLUMN].min(), (
                "Temporal leakage: train max timestamp must precede validation min timestamp."
            )
        if not df_val.empty and not df_test.empty:
            assert df_val[TIMESTAMP_COLUMN].max() < df_test[TIMESTAMP_COLUMN].min(), (
                "Temporal leakage: val max timestamp must precede test min timestamp."
            )

        partition_specs = [
            ("train", df_train, ("data/raw/shp_train.csv",), 0),
            (
                "validation",
                df_val,
                ("data/raw/shp_test.csv",),
                len([f for f in fault_timestamps if t_split_train <= f < t_split_val]),
            ),
            (
                "test",
                df_test,
                ("data/raw/shp_test.csv",),
                len([f for f in fault_timestamps if f >= t_split_val]),
            ),
        ]
        notes = (
            "Strict chronological partition: train (< 2019-02-01), validation (2019-02-01 to "
            "2019-05-01), test (>= 2019-05-01). Enforces zero temporal leakage. Fault timestamps "
            "are excluded from training and reserved strictly for retrospective evaluation."
        )
    else:
        # Published benchmark strategy: train (47,857 rows) vs test (4,897 rows)
        is_chronological = False
        partition_specs = [
            ("train", raw_train.copy(), ("data/raw/shp_train.csv",), 0),
            (
                "test",
                raw_test.copy(),
                ("data/raw/shp_test.csv",),
                len(fault_timestamps),
            ),
        ]
        notes = (
            "Published benchmark partition: replicates the original de Santis & Costa (2020) split "
            "(47,857 normal operating observations in train, 4,897 fault-adjacent observations in "
            "test) as cited in Section 4.2 and DR-05 of the project report."
        )

    for p_name, p_df, p_origins, p_faults in partition_specs:
        file_name = f"shp_{p_name}.csv"
        file_path = out_path / file_name

        p_df_export = p_df[list(REQUIRED_COLUMNS)].copy()
        # Ensure ISO formatted timestamps
        p_df_export[TIMESTAMP_COLUMN] = p_df_export[TIMESTAMP_COLUMN].dt.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        p_df_export.to_csv(file_path, index=False)

        sha256_hash = compute_file_sha256(file_path)
        t_start = (
            p_df[TIMESTAMP_COLUMN].min().isoformat()
            if not p_df.empty
            else datetime.now(UTC).isoformat()
        )
        t_end = (
            p_df[TIMESTAMP_COLUMN].max().isoformat()
            if not p_df.empty
            else datetime.now(UTC).isoformat()
        )

        parts_dict[p_name] = PartitionMetadata(
            partition_name=p_name,
            relative_path=str(file_path.as_posix()),
            record_count=len(p_df),
            start_time=t_start,
            end_time=t_end,
            sha256=sha256_hash,
            source_origins=p_origins,
            fault_events_count=p_faults,
        )

    manifest = PartitionManifest(
        strategy=strategy,
        created_at=datetime.now(UTC).isoformat(),
        is_strictly_chronological=is_chronological,
        partitions=parts_dict,
        scope_notes=notes,
    )

    manifest_path = out_path / "partitions_manifest.json"
    manifest.save(manifest_path)
    return manifest
