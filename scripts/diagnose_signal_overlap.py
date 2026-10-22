"""Descriptive follow-up to the fixed suite; no new predictor or threshold search."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.data.schema import MEASUREMENT_COLUMNS


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
    return frame


def six_hour_slope(
    timeline: pd.DataFrame, cutoff: pd.Timestamp, scale: np.ndarray
) -> np.ndarray | None:
    window = timeline[(timeline.t < cutoff) & (timeline.t >= cutoff - pd.Timedelta(hours=6))]
    if window.empty:
        return None
    age = (cutoff - window.t).dt.total_seconds().to_numpy()
    slots = np.unique(np.floor((age - 1e-9) / 300))
    if len(slots) / 72 < 0.8 or age.min() > 600:
        return None
    x = -age / 3600
    centered = x - x.mean()
    if np.dot(centered, centered) == 0:
        return None
    values = window[list(MEASUREMENT_COLUMNS)].to_numpy()
    # Fitted linear change over six hours, in training standard deviations.
    return centered @ values / np.dot(centered, centered) * 6 / scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    hashes = json.loads((args.study / "checksums.json").read_text())
    for name, expected in hashes.items():
        if compute_file_sha256(args.study / name) != expected:
            raise ValueError(f"Study evidence changed: {name}")
    study_protocol = json.loads((args.study / "protocol.json").read_text())
    if compute_file_sha256(args.inputs) != study_protocol["input_package_sha256"]:
        raise ValueError("Input package changed.")
    with zipfile.ZipFile(args.inputs) as archive:
        frames = {}
        for name in (
            "normal_training",
            "validation_context",
            "validation_nominal",
            "evaluation_faults",
        ):
            frame = pd.read_csv(io.BytesIO(archive.read(f"data/{name}.csv")))
            frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
            frames[name] = frame
    timeline = frames["validation_context"]
    nominal_times = set(frames["validation_nominal"].t)
    events = pd.read_csv(args.study / "events.csv")
    model_scores = {
        name: read_scores(args.study / f"{name}-scores.csv") for name in events.model.unique()
    }
    matched = set.intersection(*(set(scores.t) for scores in model_scores.values()))
    ranking = []
    for name, group in events.groupby("model"):
        scores = model_scores[name]
        # Use exactly the matched normal reference for all candidates.
        reference = scores[scores.t.isin(nominal_times & matched)].anomaly_score
        for event in group.to_dict("records"):
            ranking.append(
                {
                    "model": name,
                    "fault_time": event["fault_time"],
                    "eligible": event["eligible"],
                    "nominal_fraction_at_or_above_event_peak": float(
                        (reference >= event["max_score"]).mean()
                    )
                    if pd.notna(event["max_score"])
                    else None,
                }
            )
    training = frames["normal_training"]
    scale = (
        training.iloc[: int(len(training) * 0.8)][list(MEASUREMENT_COLUMNS)].std(ddof=0).to_numpy()
    )
    faults = frames["evaluation_faults"].t.tolist()
    # Descriptive controls: retain at most one eligible normal endpoint per UTC
    # hour, excluding any control window intersecting a labelled warning window.
    controls, used_hours = [], set()
    for endpoint in sorted(nominal_times):
        hour = endpoint.floor("h")
        if hour in used_hours or any(
            fault - pd.Timedelta(hours=6) <= endpoint < fault + pd.Timedelta(hours=6)
            for fault in faults
        ):
            continue
        slope = six_hour_slope(timeline, endpoint + pd.Timedelta(nanoseconds=1), scale)
        if slope is not None:
            controls.append(slope)
            used_hours.add(hour)
    reference_slopes = np.asarray(controls)
    trends = []
    for fault in faults:
        slope = six_hour_slope(timeline, fault, scale)
        for index, channel in enumerate(MEASUREMENT_COLUMNS):
            trends.append(
                {
                    "fault_time": fault.isoformat(),
                    "channel": channel,
                    "eligible": slope is not None,
                    "six_hour_change_fit_sd": float(slope[index]) if slope is not None else None,
                    "normal_absolute_slope_tail_fraction": float(
                        (np.abs(reference_slopes[:, index]) >= abs(slope[index])).mean()
                    )
                    if slope is not None and len(controls)
                    else None,
                    "normal_control_windows": len(controls),
                }
            )
    args.output.mkdir(parents=True)
    pd.DataFrame(ranking).to_csv(args.output / "event-score-ranks.csv", index=False)
    pd.DataFrame(trends).to_csv(args.output / "six-hour-trends.csv", index=False)
    pd.DataFrame(reference_slopes, columns=MEASUREMENT_COLUMNS).to_csv(
        args.output / "normal-control-trends.csv", index=False
    )
    (args.output / "notes.json").write_text(
        json.dumps(
            {
                "purpose": "Descriptive diagnosis; no fitted predictor or threshold changes",
                "controls": len(controls),
                "caveat": (
                    "Control windows overlap, are publisher selected, and are not independent. "
                    "Tail fractions are descriptive, not p-values or measured accuracy."
                ),
                "study_results_sha256": compute_file_sha256(args.study / "results.json"),
                "script_sha256": compute_file_sha256(Path(__file__)),
                "input_package_sha256": study_protocol["input_package_sha256"],
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    (args.output / "checksums.json").write_text(
        json.dumps(
            {p.name: compute_file_sha256(p) for p in args.output.iterdir() if p.is_file()}, indent=2
        )
    )
    print(f"Saved descriptive overlap diagnosis with {len(controls)} normal control windows.")


if __name__ == "__main__":
    main()
