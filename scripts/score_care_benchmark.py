"""Score saved CARE A/B model outputs with the benchmark CARE score.

Reads the per-timestamp score files written by train_care_baseline.py (v1) and
train_care_residual.py (v2); no model is refit and no threshold is changed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydropredict.analysis.care_score import care_score, score_event
from scripts.train_care_baseline import digest

MODELS = {
    "Isolation Forest (v1)": ("docs/results/care-baseline-v1", ".local/care-baseline-v1-models"),
    "Extra Trees residuals (v2)": (
        "docs/results/care-residual-v2",
        ".local/care-residual-v2-models",
    ),
}
# Reference package defaults, then the paper's wording as a sensitivity variant.
VARIANTS = {
    "reference": dict(normal_status=(0,), descend=(1, 4), threshold=72),
    "paper": dict(normal_status=(0, 2), descend=(1, 2), threshold=73),
}


def predictions(record, saved, frame, pred):
    """Pointwise threshold exceedances and the original six-reading persistent alerts."""
    if "valid" in saved:
        valid = saved.valid.to_numpy(dtype=bool)
    else:
        x = frame.iloc[pred][record["features"]].replace([np.inf, -np.inf], np.nan)
        valid = x.isna().mean(axis=1).le(0.2).to_numpy()
    pointwise = (saved.score.to_numpy() > record["threshold"]) & valid
    return {"pointwise": pointwise, "persistent": saved.alert.to_numpy(dtype=bool)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, scored = [], []
    for model, (results, scores_dir) in MODELS.items():
        records = json.loads(Path(results, "events.json").read_text())
        for record in records:
            farm, event_id = record["farm"], record["event_id"]
            path = args.root / farm / "datasets" / f"{event_id}.csv"
            assert digest(path) == record["source_sha256"], path
            columns = ["time_stamp", "train_test", "status_type_id"]
            columns += record.get("features", [])
            frame = pd.read_csv(path, sep=";", usecols=columns)
            pred = np.flatnonzero(frame.train_test.eq("prediction"))
            times = pd.DatetimeIndex(pd.to_datetime(frame.time_stamp.iloc[pred]))
            name = f"{farm.replace(' ', '_')}_{event_id}"
            saved = pd.read_csv(Path(scores_dir, f"{name}_scores.csv"), parse_dates=["time"])
            assert (pd.DatetimeIndex(saved.time) == times).all(), name
            events = pd.read_csv(args.root / farm / "event_info.csv", sep=";")
            event = events.loc[events.event_id.eq(event_id)].iloc[0]
            status = frame.status_type_id.iloc[pred].to_numpy()
            for kind, flags in predictions(record, saved, frame, pred).items():
                for variant, cfg in VARIANTS.items():
                    # Publisher guidance: Farm A status may be ignored in prediction periods.
                    normal = (
                        np.ones(len(pred), dtype=bool)
                        if farm == "Wind Farm A"
                        else np.isin(status, cfg["normal_status"])
                    )
                    result = score_event(
                        event_id,
                        event.event_label,
                        times,
                        flags,
                        normal,
                        pd.Timestamp(event.event_start),
                        pd.Timestamp(event.event_end),
                        threshold=cfg["threshold"],
                        descend=cfg["descend"],
                    )
                    key = dict(model=model, predictions=kind, variant=variant, farm=farm)
                    rows.append(key | result.__dict__)
                    scored.append((key, result))
            print(f"{model} {farm}/{event_id}: scored", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(args.output / "events.csv", index=False)
    summary = []
    for model, kind, variant in data[["model", "predictions", "variant"]].drop_duplicates().values:
        for farm in ["Wind Farm A", "Wind Farm B", "A+B"]:
            events = [
                e
                for k, e in scored
                if (k["model"], k["predictions"], k["variant"]) == (model, kind, variant)
                and farm in ("A+B", k["farm"])
            ]
            parts = care_score(events, threshold=VARIANTS[variant]["threshold"])
            key = dict(model=model, predictions=kind, variant=variant, farm=farm)
            summary.append(key | dict(events=len(events)) | parts)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    config = dict(
        script_sha256=digest(Path(__file__)),
        scorer_sha256=digest(Path("src/hydropredict/analysis/care_score.py")),
        variants=VARIANTS,
        models=MODELS,
        farm_a_status="ignored (all prediction rows treated as normal)",
        farm_c="not scored; held out",
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(pd.DataFrame(summary).to_string())


if __name__ == "__main__":
    main()
