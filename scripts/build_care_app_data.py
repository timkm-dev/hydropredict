"""Precompute replay data for the CARE early warning web app.

Uses the frozen autoencoders and decision rule; nothing is refit or retuned.
Writes frontend/care-data/fleet.json and one JSON file per monitoring period.
Outputs are derived from CC BY-SA 4.0 CARE data and stay outside Git.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydropredict.analysis.care_explain import contribution_shares
from hydropredict.analysis.care_score import criticality
from scripts.select_care_autoencoder import decide, load
from scripts.train_care_autoencoder import AutoEncoder, prepare

ALARM = 72
# Display grouping only: helps a technician scan sensors; never used in scoring.
COMPONENTS = (
    ("Transformer", ("transformer", "transformator")),
    ("Hydraulic system", ("hydraulic",)),
    ("Gearbox", ("gearbox", "gear oil")),
    ("Rotor bearing", ("rotor bearing", "main bearing")),
    ("Operating conditions", ("wind", "ambient", "outside temperature", "anemometer")),
    ("Electrical", ("power", "reactive")),
    ("Generator", ("generator", "slip ring", "stator")),
    ("Pitch and hub", ("pitch", "blade", "axis", "battery", "hub")),
    ("Converter", ("converter", "inverter", "igbt", "vcs", "vcp", "choke")),
    ("Cooling", ("cooling", "cooler", "water")),
    ("Tower and vibration", ("tower", "vibration")),
    ("Electrical", ("voltage", "current", "grid", "frequency", "consumption")),
)


def display_component(description):
    text = description.lower()
    for name, keywords in COMPONENTS:
        if any(k in text for k in keywords):
            return name
    return "Other"


def rounded(values, digits=4):
    return [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in values]


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


def alert_episodes(times, crit):
    """Contiguous runs at or above the alarm level, as index ranges."""
    above = crit >= ALARM
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2] - 1, strict=True))


def build(d, root, model_dirs, window, quantile):
    event, farm = d["event"], d["farm"]
    letter = farm[-1]
    key = f"{letter}-{int(event.event_id)}"
    folder = root / farm
    name = f"{farm.replace(' ', '_')}_{int(event.event_id)}"
    model_path = next(p / f"{name}.pt" for p in model_dirs if (p / f"{name}.pt").exists())
    times, errors, actual, expected, features = reconstruct(
        folder / "datasets" / f"{int(event.event_id)}.csv", folder, model_path
    )
    np.testing.assert_allclose(np.sqrt(errors.mean(axis=1)), d["scores"], rtol=1e-5, equal_nan=True)
    decision = decide(d, window, quantile)
    pred = d["prediction"]
    pt = decision["times"]
    crit = criticality(decision["flags"], decision["normal"])
    shares = contribution_shares(times, errors, features, window).iloc[np.flatnonzero(pred)]
    desc = pd.read_csv(folder / "feature_description.csv", sep=";", encoding="latin-1")
    info = {r.sensor_name + "_avg": r for r in desc.itertuples()}
    sensors = [
        dict(
            name=f,
            description=str(info[f].description).strip(),
            unit=str(info[f].unit).strip() if isinstance(info[f].unit, str) else "",
            component=display_component(str(info[f].description)),
        )
        for f in features
    ]
    index = {f: i for i, f in enumerate(features)}

    def top_at(row, k=3):
        values = shares.iloc[row].dropna().sort_values(ascending=False, kind="stable").head(k)
        return [[index[f], round(float(v), 3)] for f, v in values.items()]

    alerts = []
    for n, (a, b) in enumerate(alert_episodes(pt, crit), start=1):
        alerts.append(
            dict(
                id=f"{key}-{n}",
                start=pt[a].isoformat(),
                end=pt[b].isoformat(),
                peak=int(crit[a : b + 1].max()),
                top=top_at(a),
            )
        )
    frame = pd.DataFrame(
        dict(score=decision["smoothed"], crit=crit, flag=decision["flags"]), index=pt
    )
    hours = frame.groupby(pt.floor("h"))
    hourly = pd.DataFrame(
        dict(score=hours.score.last(), crit=hours.crit.last(), flag=hours.flag.mean())
    )
    last_row = pd.Series(np.arange(len(pt)), index=pt).groupby(pt.floor("h")).last()
    mean_share = shares.mean().sort_values(ascending=False)
    focus = list(
        dict.fromkeys(
            list(mean_share.index[:6]) + [features[i] for alert in alerts for i, _ in alert["top"]]
        )
    )[:10]
    series = {}
    for f in focus:
        i = index[f]
        values = pd.DataFrame(dict(a=actual[pred, i], e=expected[pred, i]), index=pt)
        grouped = values.groupby(pt.floor("h")).mean()
        series[str(i)] = dict(actual=rounded(grouped.a), expected=rounded(grouped.e))
    anomaly = event.event_label == "anomaly"
    detail = dict(
        key=key,
        farm=farm,
        asset=int(d.get("asset", -1)),
        label=event.event_label,
        fault=dict(
            start=pd.Timestamp(event.event_start).isoformat(),
            end=pd.Timestamp(event.event_end).isoformat(),
            description=str(event.event_description),
        )
        if anomaly
        else None,
        threshold=round(decision["threshold"], 5),
        alarm_level=ALARM,
        hours=[t.isoformat() for t in hourly.index],
        score=rounded(hourly.score),
        crit=[int(c) for c in hourly.crit],
        flagged=rounded(hourly.flag, 3),
        top=[top_at(int(r)) for r in last_row],
        sensors=sensors,
        series=series,
        alerts=alerts,
    )
    first = alerts[0] if alerts else None
    summary = dict(
        key=key,
        farm=farm,
        asset=detail["asset"],
        start=pt[0].isoformat(),
        end=pt[-1].isoformat(),
        recorded_fault=detail["fault"],
        max_crit=int(crit.max()),
        alerts=[
            dict(
                id=a["id"],
                start=a["start"],
                peak=a["peak"],
                top=[sensors[i]["description"] for i, _ in a["top"]],
                component=sensors[a["top"][0][0]]["component"] if a["top"] else "Other",
            )
            for a in alerts
        ],
        look_first=sensors[first["top"][0][0]]["component"] if first and first["top"] else None,
    )
    return key, detail, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scores", type=Path, required=True, help="All 95 score files")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--farms", nargs="+", default=["Wind Farm A", "Wind Farm B", "Wind Farm C"])
    parser.add_argument("--limit", type=int, help="Build only the first N periods (testing)")
    args = parser.parse_args()
    selection = json.loads((args.results / "selection.json").read_text())
    datasets = load(args.root, args.scores, args.farms)[: args.limit]
    for d in datasets:
        info = pd.read_csv(args.root / d["farm"] / "event_info.csv", sep=";")
        column = "asset" if "asset" in info else "asset_id"
        d["asset"] = int(info.loc[info.event_id.eq(d["event"].event_id), column].iloc[0])
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, started = [], time.monotonic()
    for n, d in enumerate(datasets, start=1):
        key, detail, summary = build(
            d, args.root, args.models, selection["window"], selection["quantile"]
        )
        (args.output / f"{key}.json").write_text(
            json.dumps(detail, separators=(",", ":"), allow_nan=False)
        )
        summaries.append(summary)
        elapsed = time.monotonic() - started
        print(
            f"[{n}/{len(datasets)}] {key}: {len(detail['alerts'])} alerts, "
            f"{elapsed / n * (len(datasets) - n) / 60:.0f} min left",
            flush=True,
        )
    fleet = dict(
        generated=pd.Timestamp.now("UTC").isoformat(),
        rule=dict(selection, alarm_level=ALARM, smoothing_hours=selection["window"] / 6),
        periods=sorted(summaries, key=lambda s: (-s["max_crit"], s["key"])),
    )
    (args.output / "fleet.json").write_text(json.dumps(fleet, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
