"""Timestamp-only full inventory and development-only matched precursor diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hydropredict.analysis.readiness import fault_window_coverage

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = [f"V{i}" for i in range(1, 7)]


def window_summary(frame, cutoff, scale):
    window = frame[frame.t.ge(cutoff - pd.Timedelta(hours=6)) & frame.t.lt(cutoff)]
    if not fault_window_coverage(window.t, [cutoff])[0]["eligible"]:
        return None
    early = window[window.t.lt(cutoff - pd.Timedelta(hours=5))]
    if len(early) < 9:
        return None
    x = (window.t - cutoff).dt.total_seconds().to_numpy() / 3600
    x -= x.mean()
    values = window[CHANNELS].to_numpy()
    slope = x @ values / (x @ x) * 6 / scale
    return early[CHANNELS].mean().to_numpy() / scale, slope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(exist_ok=False)
    source_hashes = {}
    for row in pd.read_csv(ROOT / "data/dataset_manifest.csv").itertuples():
        if str(row.dataset_id).startswith("shp_"):
            digest = hashlib.sha256((ROOT / row.local_filename).read_bytes()).hexdigest()
            if digest.upper() != row.sha256.upper():
                raise ValueError(f"Source changed: {row.local_filename}")
            source_hashes[row.dataset_id] = digest
    # No full-period measurement columns are loaded. Later coverage is metadata only.
    sources = {}
    for name in ("train", "test", "faults"):
        times = pd.read_csv(ROOT / f"data/raw/shp_{name}.csv", usecols=["t"])
        sources[name] = pd.to_datetime(times.t, utc=True, format="mixed").sort_values()
    times = (
        pd.concat([sources["train"], sources["test"]])
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    dt = times.diff().dt.total_seconds()
    segments = (~dt.between(150, 450)).cumsum()
    sequence_times = times[times.groupby(segments).cumcount().ge(6)]
    six_hour_times = times[
        (times - times.groupby(segments).transform("min")).ge(pd.Timedelta(hours=6))
    ]
    records = []
    group, previous = 0, None
    for fault in sources["faults"]:
        if previous is None or fault - previous > pd.Timedelta(hours=6):
            group += 1
        previous = fault
        period = (
            "earlier"
            if fault < pd.Timestamp("2019-02-01", tz="UTC")
            else "development"
            if fault < pd.Timestamp("2019-05-01", tz="UTC")
            else "later"
        )
        row = {"fault": fault.isoformat(), "period": period, "overlap_group": group}
        for kind, timestamps in (
            ("raw", times),
            ("sequence", sequence_times),
            ("six_hour_context", six_hour_times),
        ):
            info = fault_window_coverage(timestamps, [fault])[0]
            row[kind + "_eligible"] = info["eligible"]
            row[kind + "_coverage"] = info["coverage_fraction"]
        records.append(row)
    inventory = pd.DataFrame(records)
    inventory.to_csv(output / "event-inventory.csv", index=False)
    summary = []
    for period, block in inventory.groupby("period", sort=False):
        row = {
            "period": period,
            "fault_timestamps": len(block),
            "overlap_groups": block.overlap_group.nunique(),
        }
        for kind in ("raw", "sequence", "six_hour_context"):
            eligible = block[block[kind + "_eligible"]]
            row[kind + "_eligible_faults"] = len(eligible)
            row[kind + "_eligible_groups"] = eligible.overlap_group.nunique()
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "coverage-summary.csv", index=False)

    # Only the existing development package supplies measurements.
    package = ROOT / "artifacts/colab/hydropredict-lstm-input-2026-10-14.zip"
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("input_manifest.json"))
        if manifest["test_data_included"]:
            raise ValueError("Final-test measurements forbidden")
        for name, expected in manifest["files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest().upper() != expected.upper():
                raise ValueError(f"Package changed: {name}")
        frames = {}
        for name in ("normal_training", "validation_context", "evaluation_faults"):
            frame = pd.read_csv(io.BytesIO(archive.read(f"data/{name}.csv")))
            frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
            frames[name] = frame
    training = frames["normal_training"]
    scale = training.iloc[: int(len(training) * 0.8)][CHANNELS].std(ddof=0).to_numpy()
    controls = []
    previous = None
    for endpoint in training.t:
        if previous is not None and endpoint - previous < pd.Timedelta(hours=6):
            continue
        result = window_summary(training, endpoint, scale)
        if result is not None:
            controls.append((endpoint, *result))
            previous = endpoint
    baselines = np.array([c[1] for c in controls])
    trends = np.array([c[2] for c in controls])
    comparison = []
    for fault in frames["evaluation_faults"].t:
        result = window_summary(frames["validation_context"], fault, scale)
        if result is None:
            comparison.append({"fault": fault.isoformat(), "available": False})
            continue
        baseline, trend = result
        distances = np.sqrt(np.mean((baselines - baseline) ** 2, axis=1))
        nearest = np.argsort(distances, kind="stable")[:20]
        for j, channel in enumerate(CHANNELS):
            comparison.append(
                {
                    "fault": fault.isoformat(),
                    "available": True,
                    "channel": channel,
                    "six_hour_change_sd": trend[j],
                    "matched_controls": len(nearest),
                    "matched_median_distance_sd": float(np.median(distances[nearest])),
                    "matched_abs_slope_tail_fraction": float(
                        (np.abs(trends[nearest, j]) >= abs(trend[j])).mean()
                    ),
                    "all_controls_abs_slope_tail_fraction": float(
                        (np.abs(trends[:, j]) >= abs(trend[j])).mean()
                    ),
                }
            )
    pd.DataFrame(comparison).to_csv(output / "matched-development-trends.csv", index=False)
    metadata = {
        "source_hashes": source_hashes,
        "control_windows": len(controls),
        "control_selection": (
            "earlier publisher-normal windows, >=6h apart, "
            "20 nearest first-hour six-channel means in fitting-SD units"
        ),
        "physical_regime_matching": False,
        "fitted_detector": False,
        "later_measurements_read": False,
        "later_predictions_computed": False,
        "sequence_inventory": (
            "7 readings, 150–450s steps; full-union availability upper bound, not model scoring"
        ),
        "groups": "overlapping six-hour precursor windows; not verified independent failures",
        "limitations": (
            "Descriptive post-selection analysis, no p-values; control means may be poorly "
            "matched; source-normal windows are selected and calendar-shifted."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "method.json").write_text(json.dumps(metadata, indent=2))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print("Normal controls:", len(controls))
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
