"""Explain autoencoder alerts by sensor; see docs/care-explanation-protocol.md."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydropredict.analysis.care_explain import (
    at_least,
    chance_of_hit,
    contribution_shares,
    fault_component,
    sensor_component,
    top_sensors,
)
from hydropredict.data.care import care_root
from scripts.select_care_autoencoder import evaluate, load, smooth
from scripts.train_care_autoencoder import AutoEncoder, prepare


def reconstruct(path, folder, model_path):
    """Standardised inputs, reconstructions and original-unit values for scored rows."""
    data = prepare(path, folder)
    bundle = torch.load(model_path, weights_only=False)
    x = data["x"]
    assert len(bundle["mean"]) == x.shape[1], model_path
    raw = x.to_numpy(dtype=float)
    z = np.nan_to_num(((raw - bundle["mean"]) / bundle["std"]).astype(np.float32))
    model = AutoEncoder(x.shape[1])
    model.load_state_dict(bundle["state"])
    model.eval()
    span = np.concatenate([data["tail"], data["pred"]])
    with torch.no_grad():
        chunk = torch.as_tensor(z[span])
        output = model(chunk).numpy().astype(float)
    errors = (output - z[span]) ** 2
    errors[~data["valid"][span]] = np.nan
    expected = output * bundle["std"] + bundle["mean"]
    times = pd.DatetimeIndex(data["times"].iloc[span])
    return times, errors, raw[span], expected, list(x.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", help="CARE folder (default: CARE_DATA_DIR in .env)"
    )
    parser.add_argument("--farms", nargs="+", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.root = care_root(args.root)
    selection = json.loads((args.results / "selection.json").read_text())
    window, quantile = selection["window"], selection["quantile"]
    datasets = load(args.root, args.scores, args.farms)
    alarms = {(f, e.event_id): e.first_detection for f, e in evaluate(datasets, window, quantile)}
    rows = []
    for d in datasets:
        event, farm = d["event"], d["farm"]
        if event.event_label != "anomaly":
            continue
        folder = args.root / farm
        name = f"{farm.replace(' ', '_')}_{event.event_id}"
        times, errors, _, _, features = reconstruct(
            folder / "datasets" / f"{event.event_id}.csv", folder, args.scores / f"{name}.pt"
        )
        # The per-sensor errors must reproduce the saved alert scores.
        np.testing.assert_allclose(
            np.sqrt(errors.mean(axis=1)), d["scores"], rtol=1e-5, equal_nan=True
        )
        desc = pd.read_csv(folder / "feature_description.csv", sep=";", encoding="latin-1")
        describe = dict(zip(desc.sensor_name + "_avg", desc.description.str.strip(), strict=True))
        shares = contribution_shares(times, errors, features, window)
        smoothed = smooth(times, d["scores"], window)
        start, end = pd.Timestamp(event.event_start), pd.Timestamp(event.event_end)
        inside = np.flatnonzero((times >= start) & (times <= end) & ~np.isnan(smoothed))
        moments = {"alarm": alarms[(farm, int(event.event_id))]}
        moments["peak"] = times[inside[np.argmax(smoothed[inside])]] if len(inside) else None
        component = fault_component(str(event.event_description))
        matching = sum(sensor_component(describe[f]) == component for f in features)
        for moment, when in moments.items():
            if when is None:
                continue
            top = top_sensors(shares.loc[when])
            hit = any(sensor_component(describe[f]) == component for f, _ in top)
            rows.append(
                dict(
                    farm=farm,
                    event_id=int(event.event_id),
                    fault=str(event.event_description),
                    fault_component=component,
                    moment=moment,
                    time=str(when),
                    hours_before_end=(end - when).total_seconds() / 3600,
                    sensors=len(features),
                    matching_sensors=matching,
                    eligible=bool(component and matching),
                    hit=hit,
                    chance=chance_of_hit(len(features), matching) if component else None,
                )
                | {
                    f"top{i}": f"{describe[f]} ({share:.0%})"
                    for i, (f, share) in enumerate(top, start=1)
                }
            )
        print(f"{farm}/{event.event_id}: explained", flush=True)
    table = pd.DataFrame(rows)
    tag = "-".join(f[-1] for f in args.farms)
    table.to_csv(args.results / f"explanations-{tag}.csv", index=False)
    summary = []
    for moment, group in table.groupby("moment"):
        eligible = group[group.eligible]
        chances = eligible.chance.tolist()
        hits = int(eligible.hit.sum())
        summary.append(
            dict(
                moment=moment,
                explained=len(group),
                eligible=len(eligible),
                hits=hits,
                expected_by_chance=float(sum(chances)),
                p_value=at_least(chances, hits),
            )
        )
    (args.results / f"explanation-summary-{tag}.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(summary).to_string())


if __name__ == "__main__":
    main()
