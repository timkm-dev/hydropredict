"""Choose and apply the autoencoder decision rule by CARE score.

``select`` scores the protocol grid on development farms and records the best
configuration; ``confirm`` applies a recorded selection unchanged to other farms.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydropredict.analysis.care_score import care_score, score_event

WINDOWS = (1, 6, 36, 144)
QUANTILES = (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)


def smooth(times, scores, window):
    """Causal mean over the last ``window`` readings' time span; unscored stays NaN."""
    series = pd.Series(scores, index=pd.DatetimeIndex(times))
    smoothed = series.rolling(pd.Timedelta(minutes=10 * window), min_periods=1).mean()
    return np.where(np.isnan(scores), np.nan, smoothed.to_numpy())


def load(root, scores_dir, farms):
    datasets = []
    for farm in farms:
        events = pd.read_csv(root / farm / "event_info.csv", sep=";")
        for path in sorted(scores_dir.glob(f"{farm.replace(' ', '_')}_*.npz")):
            data = np.load(path)
            event_id = int(path.stem.rsplit("_", 1)[1])
            event = events.loc[events.event_id.eq(event_id)].iloc[0]
            datasets.append(dict(farm=farm, event=event, **{k: data[k] for k in data.files}))
    return datasets


def evaluate(datasets, window, quantile):
    events = []
    for d in datasets:
        times = pd.DatetimeIndex(d["times"].astype("datetime64[ns]"))
        smoothed = smooth(times, d["scores"], window)
        reference = smoothed[d["calibration"] & ~np.isnan(smoothed)]
        threshold = float(np.quantile(reference, quantile))
        pred = d["prediction"]
        flags = np.nan_to_num(smoothed[pred], nan=-np.inf) > threshold
        status = d["status"][pred]
        # Publisher guidance: Farm A status may be ignored in prediction periods.
        normal = np.ones(len(status), bool) if d["farm"] == "Wind Farm A" else status == 0
        event = d["event"]
        events.append(
            (
                d["farm"],
                score_event(
                    int(event.event_id),
                    event.event_label,
                    times[pred],
                    flags,
                    normal,
                    pd.Timestamp(event.event_start),
                    pd.Timestamp(event.event_end),
                ),
            )
        )
    return events


def summarise(events, farms, window, quantile):
    rows = []
    for farm in [*farms, "pooled"]:
        chosen = [e for f, e in events if farm in ("pooled", f)]
        if {e.label for e in chosen} == {"anomaly", "normal"}:
            rows.append(dict(window=window, quantile=quantile, farm=farm) | care_score(chosen))
    return rows


def event_table(events, window, quantile):
    return pd.DataFrame(
        [dict(farm=f, window=window, quantile=quantile) | e.__dict__ for f, e in events]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "confirm"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--farms", nargs="+", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = load(args.root, args.scores, args.farms)
    tag = "-".join(f[-1] for f in args.farms)
    selection_path = args.output / "selection.json"
    if args.mode == "select":
        assert not selection_path.exists(), "Selection already recorded; do not reselect."
        rows = []
        for window, quantile in itertools.product(WINDOWS, QUANTILES):
            rows += summarise(evaluate(datasets, window, quantile), args.farms, window, quantile)
            print(f"w={window} q={quantile}: {rows[-1]['care']:.3f}", flush=True)
        grid = pd.DataFrame(rows)
        grid.to_csv(args.output / f"grid-{tag}.csv", index=False)
        best = grid[grid.farm.eq("pooled")].sort_values("care", ascending=False).iloc[0]
        selection = dict(
            window=int(best["window"]),
            quantile=float(best["quantile"]),
            selected_on=args.farms,
            pooled_care=float(best["care"]),
            datasets=len(datasets),
        )
        selection_path.write_text(json.dumps(selection, indent=2))
    else:
        selection = json.loads(selection_path.read_text())
    window, quantile = selection["window"], selection["quantile"]
    events = evaluate(datasets, window, quantile)
    event_table(events, window, quantile).to_csv(args.output / f"events-{tag}.csv", index=False)
    summary = pd.DataFrame(summarise(events, args.farms, window, quantile))
    summary.to_csv(args.output / f"summary-{tag}.csv", index=False)
    print(json.dumps(selection), summary.to_string(), sep="\n")


if __name__ == "__main__":
    main()
