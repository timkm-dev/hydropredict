"""Dataset partitioning and chronological non-leakage contracts for HydroPredict."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
    start_time: str | None
    end_time: str | None
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
    source_accounting: dict[str, Any] = field(default_factory=dict)
    source_checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "created_at": self.created_at,
            "is_strictly_chronological": self.is_strictly_chronological,
            "partitions": {name: meta.to_dict() for name, meta in self.partitions.items()},
            "scope_notes": self.scope_notes,
            "source_accounting": self.source_accounting,
            "source_checksums": self.source_checksums,
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
            source_accounting=data.get("source_accounting", {}),
            source_checksums=data.get("source_checksums", {}),
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
    if strategy not in ("chronological", "published"):
        raise ValueError(f"Unsupported partition strategy: {strategy}")

    train_origin = str(Path(train_source).as_posix())
    test_origin = str(Path(test_source).as_posix())

    raw_train = load_telemetry_dataframe(train_source)
    raw_test = load_telemetry_dataframe(test_source)
    fault_timestamps = load_fault_log(faults_source)

    parts_dict: dict[str, PartitionMetadata] = {}
    is_chronological = False

    if strategy == "chronological":
        is_chronological = True
        t_split_train = pd.to_datetime(train_end_boundary, utc=True)
        t_split_val = pd.to_datetime(val_end_boundary, utc=True)
        if pd.isna(t_split_train) or pd.isna(t_split_val) or t_split_train >= t_split_val:
            raise ValueError("Training cutoff must precede validation cutoff.")

        # Train partition from normal operating source up to cutoff
        df_train = raw_train[raw_train[TIMESTAMP_COLUMN] < t_split_train].copy()
        # Validation partition from test source between cutoffs
        df_val = raw_test[
            (raw_test[TIMESTAMP_COLUMN] >= t_split_train)
            & (raw_test[TIMESTAMP_COLUMN] < t_split_val)
        ].copy()
        # Final test partition from test source after val cutoff
        df_test = raw_test[raw_test[TIMESTAMP_COLUMN] >= t_split_val].copy()

        if df_train.empty or df_val.empty or df_test.empty:
            raise ValueError(
                "Chronological train, validation and test partitions must be nonempty."
            )

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
            ("train", df_train, (train_origin,), 0),
            (
                "validation",
                df_val,
                (test_origin,),
                len([f for f in fault_timestamps if t_split_train <= f < t_split_val]),
            ),
            (
                "test",
                df_test,
                (test_origin,),
                len([f for f in fault_timestamps if f >= t_split_val]),
            ),
        ]
        notes = (
            f"Chronological partition: train source before {t_split_train.isoformat()}, "
            f"test source validation from {t_split_train.isoformat()} to "
            f"{t_split_val.isoformat()}, final test from {t_split_val.isoformat()}. "
            "Other source rows are excluded and counted in source_accounting. Fault counts "
            "describe cutoff intervals, not verified observable events; evaluation must assess "
            "coverage. Fault timestamps never determine training membership."
        )
    else:
        # Published benchmark strategy: train (47,857 rows) vs test (4,897 rows)
        is_chronological = False
        partition_specs = [
            ("train", raw_train.copy(), (train_origin,), 0),
            (
                "test",
                raw_test.copy(),
                (test_origin,),
                len(fault_timestamps),
            ),
        ]
        notes = (
            "Published benchmark partition: replicates the original de Santis & Costa (2020) split "
            "with overlapping calendar coverage. This optional comparison is NOT compliant "
            "with the report's DR-05 chronological evaluation requirement."
        )

    selected_train = len(partition_specs[0][1])
    selected_test = sum(len(spec[1]) for spec in partition_specs[1:])
    accounting = {
        "train_source": {
            "path": train_origin,
            "input_rows": len(raw_train),
            "selected_rows": selected_train,
            "excluded_rows": len(raw_train) - selected_train,
        },
        "test_source": {
            "path": test_origin,
            "input_rows": len(raw_test),
            "selected_rows": selected_test,
            "excluded_rows": len(raw_test) - selected_test,
        },
    }
    source_checksums = {
        str(Path(p).as_posix()): compute_file_sha256(p)
        for p in (train_source, test_source, faults_source)
    }
    out_path.mkdir(parents=True, exist_ok=True)
    for p_name, p_df, p_origins, p_faults in partition_specs:
        file_name = f"shp_{p_name}.csv"
        file_path = out_path / file_name

        p_df_export = p_df[list(REQUIRED_COLUMNS)].copy()
        # Ensure ISO formatted timestamps
        p_df_export[TIMESTAMP_COLUMN] = p_df_export[TIMESTAMP_COLUMN].map(lambda t: t.isoformat())
        p_df_export.to_csv(file_path, index=False)

        sha256_hash = compute_file_sha256(file_path)
        t_start = p_df[TIMESTAMP_COLUMN].min().isoformat() if not p_df.empty else None
        t_end = p_df[TIMESTAMP_COLUMN].max().isoformat() if not p_df.empty else None

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
        source_accounting=accounting,
        source_checksums=source_checksums,
    )

    manifest_path = out_path / "partitions_manifest.json"
    manifest.save(manifest_path)
    return manifest
