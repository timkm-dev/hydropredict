"""Fixed CARE normal-history baseline; see docs/care-baseline-protocol.md."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from hydropredict.data.care import care_root


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def persistent_alerts(times, exceed, required=6):
    times = pd.DatetimeIndex(times)
    result = np.zeros(len(times), dtype=bool)
    run = 0
    for i, value in enumerate(exceed):
        if i and (times[i] - times[i - 1]).total_seconds() != 600:
            run = 0
        run = run + 1 if value else 0
        result[i] = run >= required
    return result


def episode_starts(times, alerts):
    times = pd.DatetimeIndex(times)
    return np.array(
        [
            bool(
                a
                and (
                    i == 0 or not alerts[i - 1] or (times[i] - times[i - 1]).total_seconds() != 600
                )
            )
            for i, a in enumerate(alerts)
        ],
        dtype=bool,
    )


def run_file(path, event, features, model_dir):
    keys = ["time_stamp", "asset_id", "train_test", "status_type_id"]
    frame = pd.read_csv(
        path, sep=";", usecols=keys + features, dtype={c: "float32" for c in features}
    )
    times = pd.to_datetime(frame.time_stamp)
    assert times.is_monotonic_increasing and times.is_unique
    train_idx = np.flatnonzero(frame.train_test.eq("train"))
    pred_idx = np.flatnonzero(frame.train_test.eq("prediction"))
    assert len(train_idx) and len(pred_idx) and train_idx.max() < pred_idx.min()
    cut = int(len(train_idx) * 0.8)
    normal = frame.status_type_id.isin([0, 2]).to_numpy()
    fit_idx = train_idx[:cut][normal[train_idx[:cut]]]
    cal_idx = train_idx[cut:][normal[train_idx[cut:]]]
    x = frame[features].replace([np.inf, -np.inf], np.nan)
    kept = x.iloc[fit_idx].nunique().gt(1)
    x = x.loc[:, kept]
    valid = x.isna().mean(axis=1).le(0.2).to_numpy()
    fit_idx = fit_idx[valid[fit_idx]]
    cal_idx = cal_idx[valid[cal_idx]]
    if len(fit_idx) < 1000 or len(cal_idx) < 500 or x.shape[1] == 0:
        raise ValueError(f"Insufficient normal history: {path}")
    imputer = SimpleImputer(strategy="median").fit(x.iloc[fit_idx])
    selected = fit_idx[np.linspace(0, len(fit_idx) - 1, min(12000, len(fit_idx)), dtype=int)]
    model = IsolationForest(n_estimators=100, max_samples=256, random_state=42, n_jobs=2)
    model.fit(imputer.transform(x.iloc[selected]))
    calibration = -model.score_samples(imputer.transform(x.iloc[cal_idx]))
    threshold = float(np.quantile(calibration, 0.99))
    scores = -model.score_samples(imputer.transform(x.iloc[pred_idx]))
    pt = pd.DatetimeIndex(times.iloc[pred_idx])
    alerts = persistent_alerts(pt, (scores > threshold) & valid[pred_idx])
    starts = episode_starts(pt, alerts)
    start, end = pd.Timestamp(event.event_start), pd.Timestamp(event.event_end)
    inside = (pt >= start) & (pt <= end)
    hits = np.flatnonzero(inside & alerts)
    is_anomaly = event.event_label == "anomaly"
    lead = (end - pt[hits[0]]).total_seconds() / 3600 if len(hits) and is_anomaly else None
    name = path.parent.parent.name.replace(" ", "_") + "_" + path.stem
    model_path = model_dir / f"{name}.joblib"
    joblib.dump(
        dict(model=model, imputer=imputer, features=x.columns.tolist(), threshold=threshold),
        model_path,
        compress=3,
    )
    # Scores are local-only; public evidence contains aggregates, not telemetry.
    pd.DataFrame({"time": pt, "score": scores, "alert": alerts}).to_csv(
        model_dir / f"{name}_scores.csv", index=False
    )
    result = dict(
        farm=path.parent.parent.name,
        event_id=int(path.stem),
        asset=int(frame.asset_id.iloc[0]),
        label=event.event_label,
        description=str(event.event_description),
        source_sha256=digest(path),
        model_sha256=digest(model_path),
        features=x.columns.tolist(),
        fit_rows=len(fit_idx),
        sampled_fit_rows=len(selected),
        calibration_rows=len(cal_idx),
        fit_end=str(times.iloc[fit_idx[-1]]),
        calibration_start=str(times.iloc[cal_idx[0]]),
        calibration_end=str(times.iloc[cal_idx[-1]]),
        prediction_start=str(pt[0]),
        prediction_rows=len(pred_idx),
        valid_fraction=float(valid[pred_idx].mean()),
        threshold=threshold,
        alert_rows=int(alerts.sum()),
        episodes=int(starts.sum()),
        observed_days=len(pred_idx) / 144,
        event_rows=int(inside.sum()),
        event_alert_rows=int((inside & alerts).sum()),
        outside_event_alert_rows=int((~inside & alerts).sum()),
        detected=bool(is_anomaly and len(hits)),
        detected_24h=bool(is_anomaly and lead is not None and lead >= 24),
        first_event_alert=str(pt[hits[0]]) if is_anomaly and len(hits) else None,
        end_relative_lead_hours=lead,
        preexisting_alert=bool(
            is_anomaly and len(hits) and hits[0] > 0 and alerts[hits[0] - 1] and not starts[hits[0]]
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", help="CARE folder (default: CARE_DATA_DIR in .env)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    args.root = care_root(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(
        protocol_sha256=digest(Path("docs/care-baseline-protocol.md")),
        script_sha256=digest(Path(__file__)),
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        seed=42,
        model="IsolationForest",
        trees=100,
        max_samples=256,
        calibration_quantile=0.99,
        persistence_readings=6,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for farm in ["Wind Farm A", "Wind Farm B"]:
        folder = args.root / farm
        desc = pd.read_csv(folder / "feature_description.csv", sep=";")
        allowed = desc.loc[
            ~desc.is_angle.astype(str).str.lower().eq("true")
            & ~desc.is_counter.astype(str).str.lower().eq("true"),
            "sensor_name",
        ]
        events = pd.read_csv(folder / "event_info.csv", sep=";")
        for path in sorted((folder / "datasets").glob("*.csv"), key=lambda p: int(p.stem)):
            header = pd.read_csv(path, sep=";", nrows=0).columns
            features = [c for c in header if c.endswith("_avg") and c[:-4] in set(allowed)]
            event = events.loc[events.event_id.eq(int(path.stem))].iloc[0]
            row = run_file(path, event, features, args.models)
            rows.append(row)
            (args.output / "events.json").write_text(json.dumps(rows, indent=2))
            print(f"{farm}/{path.stem}: completed", flush=True)
    data = pd.DataFrame(rows)
    summary = []
    for farm, group in data.groupby("farm"):
        anomalies = group[group.label.eq("anomaly")]
        controls = group[group.label.eq("normal")]
        rate = controls.episodes.sum() / controls.observed_days.sum() * 30
        fraction = controls.alert_rows.sum() / controls.prediction_rows.sum()
        recall = anomalies.detected_24h.mean()
        summary.append(
            dict(
                farm=farm,
                anomaly_files=len(anomalies),
                detected=int(anomalies.detected.sum()),
                detected_24h=int(anomalies.detected_24h.sum()),
                normal_files=len(controls),
                normal_files_with_alerts=int(controls.episodes.gt(0).sum()),
                normal_episodes=int(controls.episodes.sum()),
                normal_observed_days=controls.observed_days.sum(),
                normal_episodes_per_30_days=rate,
                normal_alert_fraction=fraction,
                minimum_valid_fraction=group.valid_fraction.min(),
                gate_pass=bool(
                    recall >= 0.5
                    and rate <= 1
                    and fraction <= 0.01
                    and group.valid_fraction.min() >= 0.8
                ),
            )
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    data.drop(columns=["features"]).to_csv(args.output / "events.csv", index=False)
    data.groupby(["farm", "asset", "label"]).agg(
        files=("event_id", "size"),
        detections_24h=("detected_24h", "sum"),
        episodes=("episodes", "sum"),
        rows=("prediction_rows", "sum"),
        alert_rows=("alert_rows", "sum"),
    ).to_csv(args.output / "turbines.csv")
    config["elapsed_seconds"] = time.monotonic() - started
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
