"""Nested turbine-held-out CARE classification of causal residual trajectories."""

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scripts.train_care_baseline import digest, episode_starts, persistent_alerts
from scripts.train_care_residual import context_features


def trajectories(times, residuals):
    groups = pd.Series(pd.DatetimeIndex(times)).diff().ne(pd.Timedelta(minutes=10)).cumsum()
    means = []
    for window in [6, 36]:
        means.append(
            residuals.groupby(groups.to_numpy(), group_keys=False).transform(
                lambda x, window=window: x.rolling(window, min_periods=window).mean()
            )
        )
    return pd.concat(
        [
            residuals,
            means[0].add_suffix("_mean6"),
            means[1].add_suffix("_mean36"),
            (residuals - means[1]).add_suffix("_departure"),
        ],
        axis=1,
    )


def load_cases(root, model_root):
    previous = json.loads(Path("docs/results/care-residual-v2/events.json").read_text())
    cases = []
    for record in previous:
        farm, event_id = record["farm"], record["event_id"]
        source = root / farm / "datasets" / f"{event_id}.csv"
        model_path = model_root / f"{farm.replace(' ', '_')}_{event_id}.joblib"
        assert digest(source) == record["source_sha256"]
        assert digest(model_path) == record["model_sha256"]
        bundle = joblib.load(model_path)
        context, targets = bundle["context"], bundle["targets"]
        frame = pd.read_csv(
            source,
            sep=";",
            usecols=["time_stamp", "train_test"] + context + targets,
            dtype={c: "float32" for c in context + targets},
        )
        times = pd.to_datetime(frame.time_stamp)
        raw = frame[context + targets].replace([np.inf, -np.inf], np.nan)
        x = context_features(times, raw[context])
        idx = np.flatnonzero(frame.train_test.eq("prediction"))
        valid = (
            x.iloc[idx].notna().all(axis=1).to_numpy()
            & raw[targets].iloc[idx].notna().all(axis=1).to_numpy()
        )
        z = np.full((len(idx), len(targets)), np.nan)
        z[valid] = (
            (raw[targets].iloc[idx[valid]].to_numpy() - bundle["model"].predict(x.iloc[idx[valid]]))
            - bundle["center"]
        ) / bundle["scale"]
        pt = pd.DatetimeIndex(times.iloc[idx])
        features = trajectories(pt, pd.DataFrame(z, columns=targets)).astype("float32")
        info = pd.read_csv(root / farm / "event_info.csv", sep=";")
        event = info[info.event_id.eq(event_id)].iloc[0]
        start, end = pd.Timestamp(event.event_start), pd.Timestamp(event.event_end)
        inside = (pt >= start) & (pt <= end)
        eligible = features.notna().all(axis=1).to_numpy()
        positive = inside & (pt <= end - pd.Timedelta(hours=24))
        fitting = eligible & (positive if record["label"] == "anomaly" else True)
        indices = np.flatnonzero(fitting)
        if len(indices):
            indices = indices[np.linspace(0, len(indices) - 1, min(500, len(indices)), dtype=int)]
        cases.append(
            dict(
                farm=farm,
                event_id=event_id,
                asset=record["asset"],
                label=record["label"],
                times=pt,
                x=features,
                valid=eligible,
                fit_idx=indices,
                inside=inside,
                end=end,
                source_sha256=record["source_sha256"],
            )
        )
        print(f"Prepared {farm}/{event_id}", flush=True)
    return cases


def train(cases, excluded):
    chosen = [c for c in cases if c["asset"] not in excluded and len(c["fit_idx"])]
    assert all(c["asset"] not in excluded for c in chosen)
    counts = {label: sum(c["label"] == label for c in chosen) for label in ["anomaly", "normal"]}
    if not all(counts.values()):
        raise ValueError("Both classes required in every grouped fit")
    x, y, w = [], [], []
    for case in chosen:
        part = case["x"].iloc[case["fit_idx"]]
        x.append(part)
        y.extend([int(case["label"] == "anomaly")] * len(part))
        w.extend([1 / (counts[case["label"]] * len(part))] * len(part))
    weights = np.asarray(w)
    weights *= len(weights) / weights.sum()
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=10,
        early_stopping=False,
        random_state=42,
    )
    model.fit(pd.concat(x, ignore_index=True), np.asarray(y), sample_weight=weights)
    return model


def predict(model, case):
    result = np.full(len(case["x"]), np.nan)
    result[case["valid"]] = model.predict_proba(case["x"].loc[case["valid"]])[:, 1]
    return result


def calibrate(control_scores):
    audit = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0]:
        total_rows = episodes = alert_rows = 0
        for case, score in control_scores:
            assert case["label"] == "normal"
            alerts = persistent_alerts(case["times"], score > threshold)
            episodes += int(episode_starts(case["times"], alerts).sum())
            alert_rows += int(alerts.sum())
            total_rows += len(score)
        rate = episodes / (total_rows / 144) * 30
        fraction = alert_rows / total_rows
        audit.append(dict(threshold=threshold, episodes_per_30_days=rate, alert_fraction=fraction))
        if rate <= 1 and fraction <= 0.01:
            return threshold, audit
    raise AssertionError("No-alarm fallback must pass")


def evaluate(case, score, threshold):
    alerts = persistent_alerts(case["times"], score > threshold)
    starts = episode_starts(case["times"], alerts)
    hits = np.flatnonzero(alerts & case["inside"])
    anomaly = case["label"] == "anomaly"
    lead = (
        (case["end"] - case["times"][hits[0]]).total_seconds() / 3600
        if anomaly and len(hits)
        else None
    )
    return dict(
        farm=case["farm"],
        event_id=case["event_id"],
        asset=case["asset"],
        label=case["label"],
        source_sha256=case["source_sha256"],
        threshold=threshold,
        prediction_rows=len(score),
        valid_fraction=float(case["valid"].mean()),
        detected=bool(anomaly and len(hits)),
        detected_24h=bool(lead is not None and lead >= 24),
        end_relative_lead_hours=lead,
        first_event_alert=str(case["times"][hits[0]]) if anomaly and len(hits) else None,
        preexisting_alert=bool(
            anomaly and len(hits) and hits[0] > 0 and alerts[hits[0] - 1] and not starts[hits[0]]
        ),
        alert_rows=int(alerts.sum()),
        episodes=int(starts.sum()),
        observed_days=len(score) / 144,
    )


def summarize(data):
    summaries = []
    rng = np.random.default_rng(42)
    for farm, g in data.groupby("farm"):
        a = g[g.label.eq("anomaly")]
        n = g[g.label.eq("normal")]
        rate = float(n.episodes.sum() / n.observed_days.sum() * 30)
        fraction = float(n.alert_rows.sum() / n.prediction_rows.sum())
        detections = []
        rates = []
        assets = g.asset.unique()
        for _ in range(2000):
            resample = pd.concat(
                [g[g.asset.eq(k)] for k in rng.choice(assets, len(assets), replace=True)]
            )
            aa = resample[resample.label.eq("anomaly")]
            nn = resample[resample.label.eq("normal")]
            if len(aa):
                detections.append(float(aa.detected_24h.mean()))
            if len(nn):
                rates.append(float(nn.episodes.sum() / nn.observed_days.sum() * 30))
        summaries.append(
            dict(
                farm=farm,
                anomaly_files=len(a),
                detected=int(a.detected.sum()),
                detected_24h=int(a.detected_24h.sum()),
                normal_episodes=int(n.episodes.sum()),
                normal_episodes_per_30_days=rate,
                normal_alert_fraction=fraction,
                minimum_valid_fraction=float(g.valid_fraction.min()),
                detection_cluster_interval=np.quantile(detections, [0.025, 0.975]).tolist(),
                alarm_cluster_interval=np.quantile(rates, [0.025, 0.975]).tolist(),
                undefined_detection_resamples=2000 - len(detections),
                undefined_alarm_resamples=2000 - len(rates),
                gate_pass=bool(
                    a.detected_24h.mean() >= 0.5
                    and rate <= 1
                    and fraction <= 0.01
                    and g.valid_fraction.min() >= 0.8
                ),
            )
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(
            dict(
                protocol_sha256=digest(Path("docs/care-supervised-protocol.md")),
                script_sha256=digest(Path(__file__)),
            ),
            indent=2,
        )
    )
    cases = load_cases(args.root, Path(".local/care-residual-v2-models"))
    rows = []
    folds = []
    for farm in sorted({c["farm"] for c in cases}):
        local = [c for c in cases if c["farm"] == farm]
        assert all(c["x"].columns.equals(local[0]["x"].columns) for c in local)
        assets = sorted({c["asset"] for c in local})
        pair_scores = {}
        for left, right in itertools.combinations(assets, 2):
            model = train(local, {left, right})
            for c in local:
                if c["asset"] in {left, right} and c["label"] == "normal":
                    pair_scores[(left, right, c["event_id"])] = predict(model, c)
        for held in assets:
            controls = []
            for c in local:
                if c["asset"] != held and c["label"] == "normal":
                    left, right = sorted([held, c["asset"]])
                    controls.append((c, pair_scores[(left, right, c["event_id"])]))
            threshold, audit = calibrate(controls)
            model = train(local, {held})
            model_path = args.models / f"{farm.replace(' ', '_')}_{held}.joblib"
            joblib.dump(
                dict(model=model, threshold=threshold, features=local[0]["x"].columns.tolist()),
                model_path,
                compress=3,
            )
            folds.append(
                dict(
                    farm=farm,
                    held_asset=held,
                    training_assets=[a for a in assets if a != held],
                    calibration_assets=sorted({c["asset"] for c, _ in controls}),
                    threshold=threshold,
                    threshold_audit=audit,
                    model_sha256=digest(model_path),
                )
            )
            for c in local:
                if c["asset"] == held:
                    score = predict(model, c)
                    rows.append(evaluate(c, score, threshold))
                    pd.DataFrame(dict(time=c["times"], score=score, valid=c["valid"])).to_csv(
                        args.models / f"{farm.replace(' ', '_')}_{c['event_id']}_scores.csv",
                        index=False,
                    )
            (args.output / "events.json").write_text(json.dumps(rows, indent=2))
            (args.output / "folds.json").write_text(json.dumps(folds, indent=2))
            print(f"{farm}: held-out turbine {held} completed", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(args.output / "events.csv", index=False)
    summary = summarize(data)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
