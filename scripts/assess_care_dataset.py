"""Read-only, bounded-memory CARE inventory; writes aggregate evidence only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    inventory, quality, metadata = [], [], []
    for farm in sorted(args.root.glob("Wind Farm *")):
        events = pd.read_csv(farm / "event_info.csv", sep=";")
        metadata.append(
            {
                "farm": farm.name,
                "labels": events.event_label.value_counts().to_dict(),
                "metadata_sha256": hashlib.sha256(
                    (farm / "event_info.csv").read_bytes()
                ).hexdigest(),
            }
        )
        for path in sorted((farm / "datasets").glob("*.csv"), key=lambda p: int(p.stem)):
            event = events.loc[events.event_id == int(path.stem)].iloc[0]
            header = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
            sensors = [c for c in header if c.endswith("_avg")]
            keys = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
            record = {
                "farm": farm.name,
                "event_id": int(path.stem),
                "label": event.event_label,
                "bytes": path.stat().st_size,
                "columns": len(header),
                "avg_features": len(sensors),
                "rows": 0,
                "bad_steps": 0,
                "duplicate_times": 0,
                "event_rows": 0,
                "normal_train_rows": 0,
            }
            counts, states, assets, seen = {}, {}, set(), set()
            missing = pd.Series(0, index=sensors, dtype="int64")
            low = pd.Series(np.inf, index=sensors)
            high = pd.Series(-np.inf, index=sensors)
            last = None
            bounds = {}
            start, end = pd.to_datetime(event.event_start), pd.to_datetime(event.event_end)
            for chunk in pd.read_csv(
                path,
                sep=";",
                usecols=keys + sensors,
                chunksize=4096,
                dtype={c: "float32" for c in sensors},
            ):
                ts = pd.to_datetime(chunk.time_stamp, errors="raise")
                delta = ts.diff().dt.total_seconds()
                if last is not None:
                    delta.iloc[0] = (ts.iloc[0] - last).total_seconds()
                record["bad_steps"] += int((delta.notna() & delta.ne(600)).sum())
                last = ts.iloc[-1]
                values = ts.astype("int64").tolist()
                record["duplicate_times"] += (
                    len(values) - len(set(values)) + len(set(values) & seen)
                )
                seen.update(values)
                assets.update(chunk.asset_id.unique().tolist())
                record["rows"] += len(chunk)
                record["event_rows"] += int(ts.between(start, end).sum())
                record["normal_train_rows"] += int(
                    (chunk.train_test.eq("train") & chunk.status_type_id.isin([0, 2])).sum()
                )
                for key, group in chunk.groupby("train_test"):
                    counts[key] = counts.get(key, 0) + len(group)
                    times = pd.to_datetime(group.time_stamp)
                    old = bounds.get(key, [times.min(), times.max()])
                    bounds[key] = [min(old[0], times.min()), max(old[1], times.max())]
                for key, value in chunk.status_type_id.value_counts().items():
                    states[str(key)] = states.get(str(key), 0) + int(value)
                x = chunk[sensors].replace([np.inf, -np.inf], np.nan)
                missing += x.isna().sum()
                low = pd.concat([low, x.min()], axis=1).min(axis=1)
                high = pd.concat([high, x.max()], axis=1).max(axis=1)
            record.update(
                assets=sorted(assets),
                split_counts=counts,
                status_counts=states,
                split_bounds={k: [str(t) for t in v] for k, v in bounds.items()},
                event_start=str(start),
                event_end=str(end),
                event_duration_days=(end - start).total_seconds() / 86400,
                avg_missing_fraction=float(missing.sum() / (record["rows"] * len(sensors))),
                constant_avg_features=int((low == high).sum()),
                all_missing_avg_features=int((missing == record["rows"]).sum()),
            )
            inventory.append(record)
            for c in sensors:
                quality.append(
                    {
                        "farm": farm.name,
                        "event_id": int(path.stem),
                        "feature": c,
                        "missing_fraction": float(missing[c] / record["rows"]),
                        "constant": bool(low[c] == high[c]),
                    }
                )
            (args.output / "inventory.json").write_text(
                json.dumps(inventory, indent=2), encoding="utf-8"
            )
            print(
                f"{farm.name}/{path.stem}: {record['rows']:,} rows; "
                f"missing {record['avg_missing_fraction']:.2%}",
                flush=True,
            )
    pd.DataFrame(quality).to_csv(args.output / "feature_quality.csv", index=False)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.monotonic() - started,
                "chunk_rows": 4096,
                "scope": "All rows, metadata and average channels only; no model fitting",
                "farms": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
