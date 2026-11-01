"""Fixed conditional temperature monitoring experiment for CARE A/B."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from scripts.train_care_baseline import digest, episode_starts, persistent_alerts

CONTEXT = {
    "Wind Farm A": ["sensor_0_avg", "wind_speed_3_avg", "power_30_avg", "sensor_52_avg"],
    "Wind Farm B": ["sensor_8_avg", "wind_speed_61_avg", "power_62_avg", "sensor_25_avg"],
}


def context_features(times, values):
    """Current context plus six-point causal means, with gaps resetting history."""
    group = pd.Series(pd.DatetimeIndex(times)).diff().ne(pd.Timedelta(minutes=10)).cumsum()
    means = values.groupby(group.to_numpy(), group_keys=False).transform(
        lambda x: x.rolling(6, min_periods=6).mean()
    )
    return pd.concat([values, means.add_suffix("_mean6")], axis=1)


def fit_monitor(frame, context, targets):
    times = pd.to_datetime(frame.time_stamp)
    train = np.flatnonzero(frame.train_test.eq("train"))
    assert times.is_monotonic_increasing and times.is_unique
    pred = np.flatnonzero(frame.train_test.eq("prediction"))
    assert train[-1] < pred[0]
    normal = frame.status_type_id.isin([0, 2]).to_numpy()
    first, second = int(len(train) * 0.8), int(len(train) * 0.9)
    fit = train[:first][normal[train[:first]]]
    raw = frame[context + targets].replace([np.inf, -np.inf], np.nan)
    targets = [c for c in targets if raw.iloc[fit][c].nunique() > 1]
    x = context_features(times, raw[context])
    y = raw[targets]
    valid = x.notna().all(axis=1).to_numpy() & y.notna().all(axis=1).to_numpy()
    fit = fit[valid[fit]]
    scale_idx = train[first:second]
    scale_idx = scale_idx[normal[scale_idx] & valid[scale_idx]]
    threshold_idx = train[second:]
    threshold_idx = threshold_idx[normal[threshold_idx] & valid[threshold_idx]]
    if len(fit) < 1000 or len(scale_idx) < 250 or len(threshold_idx) < 250 or not targets:
        raise ValueError("Insufficient eligible training history")
    selected = fit[np.linspace(0, len(fit) - 1, min(len(fit), 12000), dtype=int)]
    model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        max_features=1.0,
        random_state=42,
        n_jobs=2,
    )
    fit_y = y.iloc[selected].to_numpy()
    model.fit(x.iloc[selected], fit_y.ravel() if len(targets) == 1 else fit_y)
    residual = y.iloc[scale_idx].to_numpy() - model.predict(x.iloc[scale_idx]).reshape(
        -1, len(targets)
    )
    center = np.median(residual, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(residual - center), axis=0), 1.0)

    def scores(indices):
        result = np.full(len(indices), np.nan)
        good = valid[indices]
        if not good.any():
            return result
        residual = y.iloc[indices[good]].to_numpy() - model.predict(x.iloc[indices[good]]).reshape(
            -1, len(targets)
        )
        result[good] = np.max(np.abs((residual - center) / scale), axis=1)
        return result

    threshold = float(np.quantile(scores(threshold_idx), 0.99))
    bundle = dict(
        model=model,
        context=context,
        targets=targets,
        center=center,
        scale=scale,
        threshold=threshold,
        window=6,
    )
    audit = dict(
        fit_rows=len(fit),
        scale_rows=len(scale_idx),
        calibration_rows=len(threshold_idx),
        fit_end=str(times.iloc[fit[-1]]),
        scale_start=str(times.iloc[scale_idx[0]]),
        scale_end=str(times.iloc[scale_idx[-1]]),
        calibration_start=str(times.iloc[threshold_idx[0]]),
        calibration_end=str(times.iloc[threshold_idx[-1]]),
    )
    return bundle, scores(pred), valid[pred], pred, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(
        protocol_sha256=digest(Path("docs/care-residual-protocol.md")),
        script_sha256=digest(Path(__file__)),
        sklearn=sklearn.__version__,
        pandas=pd.__version__,
        numpy=np.__version__,
        model="ExtraTreesRegressor",
        trees=100,
        max_depth=12,
        min_samples_leaf=20,
        seed=42,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for farm, context in CONTEXT.items():
        folder = args.root / farm
        desc = pd.read_csv(folder / "feature_description.csv", sep=";")
        targets = [
            r.sensor_name + "_avg"
            for r in desc.itertuples()
            if "temperature" in str(r.description).lower() and r.sensor_name + "_avg" not in context
        ]
        events = pd.read_csv(folder / "event_info.csv", sep=";")
        for path in sorted((folder / "datasets").glob("*.csv"), key=lambda p: int(p.stem)):
            event = events.loc[events.event_id.eq(int(path.stem))].iloc[0]
            frame = pd.read_csv(
                path,
                sep=";",
                usecols=["time_stamp", "asset_id", "train_test", "status_type_id"]
                + context
                + targets,
                dtype={c: "float32" for c in context + targets},
            )
            bundle, scores, valid, pred, audit = fit_monitor(frame, context, targets)
            times = pd.DatetimeIndex(pd.to_datetime(frame.time_stamp.iloc[pred]))
            alerts = persistent_alerts(times, (scores > bundle["threshold"]) & valid)
            starts = episode_starts(times, alerts)
            start, end = pd.Timestamp(event.event_start), pd.Timestamp(event.event_end)
            inside = (times >= start) & (times <= end)
            hits = np.flatnonzero(inside & alerts)
            anomaly = event.event_label == "anomaly"
            lead = (end - times[hits[0]]).total_seconds() / 3600 if anomaly and len(hits) else None
            name = f"{farm.replace(' ', '_')}_{path.stem}"
            model_path = args.models / f"{name}.joblib"
            joblib.dump(bundle, model_path, compress=3)
            pd.DataFrame(dict(time=times, score=scores, alert=alerts, valid=valid)).to_csv(
                args.models / f"{name}_scores.csv", index=False
            )
            row = dict(
                farm=farm,
                event_id=int(path.stem),
                asset=int(frame.asset_id.iloc[0]),
                label=event.event_label,
                description=str(event.event_description),
                source_sha256=digest(path),
                model_sha256=digest(model_path),
                targets=bundle["targets"],
                threshold=bundle["threshold"],
                **audit,
                prediction_start=str(times[0]),
                prediction_rows=len(pred),
                valid_fraction=float(valid.mean()),
                alert_rows=int(alerts.sum()),
                episodes=int(starts.sum()),
                observed_days=len(pred) / 144,
                event_rows=int(inside.sum()),
                event_alert_rows=int((inside & alerts).sum()),
                detected=bool(anomaly and len(hits)),
                detected_24h=bool(lead is not None and lead >= 24),
                first_event_alert=str(times[hits[0]]) if anomaly and len(hits) else None,
                first_alert_status=int(frame.status_type_id.iloc[pred[hits[0]]])
                if anomaly and len(hits)
                else None,
                end_relative_lead_hours=lead,
                preexisting_alert=bool(
                    anomaly
                    and len(hits)
                    and hits[0] > 0
                    and alerts[hits[0] - 1]
                    and not starts[hits[0]]
                ),
            )
            rows.append(row)
            (args.output / "events.json").write_text(json.dumps(rows, indent=2))
            print(f"{farm}/{path.stem}: completed", flush=True)
    data = pd.DataFrame(rows)
    data.drop(columns="targets").to_csv(args.output / "events.csv", index=False)
    summary = []
    for farm, g in data.groupby("farm"):
        a = g[g.label.eq("anomaly")]
        n = g[g.label.eq("normal")]
        rate = float(n.episodes.sum() / n.observed_days.sum() * 30)
        fraction = float(n.alert_rows.sum() / n.prediction_rows.sum())
        summary.append(
            dict(
                farm=farm,
                anomaly_files=len(a),
                detected=int(a.detected.sum()),
                detected_24h=int(a.detected_24h.sum()),
                normal_episodes=int(n.episodes.sum()),
                normal_files_with_alerts=int(n.episodes.gt(0).sum()),
                normal_observed_days=float(n.observed_days.sum()),
                normal_episodes_per_30_days=rate,
                normal_alert_fraction=fraction,
                minimum_valid_fraction=float(g.valid_fraction.min()),
                gate_pass=bool(
                    a.detected_24h.mean() >= 0.5
                    and rate <= 1
                    and fraction <= 0.01
                    and g.valid_fraction.min() >= 0.8
                ),
            )
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    data.groupby(["farm", "asset", "label"]).agg(
        files=("event_id", "size"),
        detected_24h=("detected_24h", "sum"),
        episodes=("episodes", "sum"),
        rows=("prediction_rows", "sum"),
        alert_rows=("alert_rows", "sum"),
    ).to_csv(args.output / "turbines.csv")
    config["elapsed_seconds"] = time.monotonic() - started
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
