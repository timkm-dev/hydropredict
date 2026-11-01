"""Fixed diverse classifier comparison on the frozen v3 CARE representations."""

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.train_care_baseline import digest

FAMILIES = ["logistic", "gaussian_nb", "random_forest", "rbf_svm_approx"]


def fast_alerts(times, exceed, required=6):
    exceed = np.asarray(exceed, dtype=bool)
    if not len(exceed):
        return exceed.copy(), exceed.copy()
    stamps = pd.DatetimeIndex(times).to_numpy(dtype="datetime64[ns]").astype("int64")
    gap = np.r_[True, np.diff(stamps) != 600_000_000_000]
    index = np.arange(len(exceed))
    reset = np.maximum.accumulate(np.where(~exceed, index, np.where(gap, index - 1, -1)))
    alerts = index - reset >= required
    starts = alerts & (gap | ~np.r_[False, alerts[:-1]])
    return alerts, starts


def make_model(family, dimension):
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        )
    if family == "logistic":
        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000)
    elif family == "gaussian_nb":
        clf = GaussianNB(var_smoothing=0.01)
    elif family == "rbf_svm_approx":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "kernel",
                    Nystroem(kernel="rbf", gamma=1 / dimension, n_components=128, random_state=42),
                ),
                ("clf", LinearSVC(C=0.1, dual=False, max_iter=5000, random_state=42)),
            ]
        )
    else:
        raise ValueError(family)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def fitting_data(cases, excluded):
    chosen = [c for c in cases if c["asset"] not in excluded and len(c["fit_idx"])]
    counts = {label: sum(c["label"] == label for c in chosen) for label in ["normal", "anomaly"]}
    assert all(counts.values())
    x = []
    y = []
    w = []
    for c in chosen:
        part = c["x"].iloc[c["fit_idx"]]
        x.append(part)
        y.extend([int(c["label"] == "anomaly")] * len(part))
        w.extend([1 / (counts[c["label"]] * len(part))] * len(part))
    weights = np.asarray(w)
    weights *= len(weights) / weights.sum()
    return pd.concat(x, ignore_index=True), np.asarray(y), weights


def fit(cases, excluded, family):
    x, y, w = fitting_data(cases, excluded)
    model = make_model(family, x.shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if isinstance(model, Pipeline):
            model.fit(x, y, scale__sample_weight=w, clf__sample_weight=w)
        else:
            model.fit(x, y, sample_weight=w)
    return model, [str(v.message) for v in caught]


def score_values(model, x, family):
    if family == "rbf_svm_approx":
        return expit(model.decision_function(x))
    return model.predict_proba(x)[:, 1]


def predict(model, case, family):
    score = np.full(len(case["x"]), np.nan)
    score[case["valid"]] = score_values(model, case["x"].loc[case["valid"]], family)
    return score


def calibrate(controls):
    audit = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0]:
        rows = episodes = alert_rows = 0
        for case, score in controls:
            assert case["label"] == "normal"
            alerts, starts = fast_alerts(case["times"], score > threshold)
            episodes += int(starts.sum())
            alert_rows += int(alerts.sum())
            rows += len(score)
        rate = episodes / (rows / 144) * 30
        fraction = alert_rows / rows
        audit.append(dict(threshold=threshold, episodes_per_30_days=rate, alert_fraction=fraction))
        if rate <= 1 and fraction <= 0.01:
            return threshold, audit
    raise AssertionError("No-alarm fallback failed")


def evaluate(case, score, threshold):
    alerts, starts = fast_alerts(case["times"], score > threshold)
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


def auc_diagnostics(model, cases, held, family):
    result = {}
    for stage in ["fitting", "held_out"]:
        y = []
        scores = []
        weights = []
        for c in cases:
            if (c["asset"] == held) != (stage == "held_out") or not len(c["fit_idx"]):
                continue
            part = c["x"].iloc[c["fit_idx"]]
            y.extend([int(c["label"] == "anomaly")] * len(part))
            scores.extend(score_values(model, part, family))
            weights.extend([1 / len(part)] * len(part))
        result[stage + "_auc"] = (
            float(roc_auc_score(y, scores, sample_weight=weights)) if len(set(y)) == 2 else None
        )
    return result


def summarize(data):
    result = []
    for (family, farm), g in data.groupby(["family", "farm"]):
        a = g[g.label.eq("anomaly")]
        n = g[g.label.eq("normal")]
        rate = float(n.episodes.sum() / n.observed_days.sum() * 30)
        fraction = float(n.alert_rows.sum() / n.prediction_rows.sum())
        result.append(
            dict(
                family=family,
                farm=farm,
                anomaly_files=len(a),
                detected=int(a.detected.sum()),
                detected_24h=int(a.detected_24h.sum()),
                normal_episodes=int(n.episodes.sum()),
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
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=False)
    cache = Path(".local/care-supervised-v3-diagnostic-features.joblib")
    identity = json.loads(cache.with_suffix(".json").read_text())
    assert identity == dict(
        upstream=digest(Path("docs/results/care-residual-v2/events.json")),
        feature_code=digest(Path("scripts/research_care_supervised.py")),
    )
    cases = joblib.load(cache)
    reference = pd.read_csv("docs/results/care-supervised-v3/events.csv")
    checked = 0
    for case in cases:
        prior = reference[
            (reference.farm == case["farm"]) & (reference.event_id == case["event_id"])
        ].iloc[0]
        assert prior.source_sha256 == case["source_sha256"]
        scores_path = Path(".local/care-supervised-v3-models") / (
            f"{case['farm'].replace(' ', '_')}_{case['event_id']}_scores.csv"
        )
        old_scores = pd.read_csv(scores_path).score.to_numpy()
        replay = evaluate(case, old_scores, float(prior.threshold))
        for field in ["prediction_rows", "alert_rows", "episodes", "detected", "detected_24h"]:
            assert replay[field] == prior[field], (case["event_id"], field)
        checked += 1
    assert checked == 37
    config = dict(
        protocol_sha256=digest(Path("docs/care-model-comparison-protocol.md")),
        code_sha256=digest(Path(__file__)),
        cache_sha256=digest(cache),
        cache_identity=identity,
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        families=FAMILIES,
        vectorized_replay_cases_verified=checked,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    folds = []
    fit_warnings = []
    timings = []
    for family in FAMILIES:
        began = time.monotonic()
        for farm in sorted({c["farm"] for c in cases}):
            local = [c for c in cases if c["farm"] == farm]
            assets = sorted({c["asset"] for c in local})
            pairs = {}
            for left, right in itertools.combinations(assets, 2):
                model, messages = fit(local, {left, right}, family)
                if messages:
                    fit_warnings.append(
                        dict(family=family, farm=farm, excluded=[left, right], messages=messages)
                    )
                for c in local:
                    if c["asset"] in {left, right} and c["label"] == "normal":
                        pairs[(left, right, c["event_id"])] = predict(model, c, family)
            for held in assets:
                controls = []
                for c in local:
                    if c["asset"] != held and c["label"] == "normal":
                        left, right = sorted([held, c["asset"]])
                        controls.append((c, pairs[(left, right, c["event_id"])]))
                threshold, audit = calibrate(controls)
                model, messages = fit(local, {held}, family)
                if messages:
                    fit_warnings.append(
                        dict(family=family, farm=farm, excluded=[held], messages=messages)
                    )
                model_path = args.models / f"{family}_{farm.replace(' ', '_')}_{held}.joblib"
                joblib.dump(
                    dict(
                        model=model,
                        threshold=threshold,
                        family=family,
                        features=local[0]["x"].columns.tolist(),
                    ),
                    model_path,
                    compress=3,
                )
                folds.append(
                    dict(
                        family=family,
                        farm=farm,
                        held_asset=held,
                        training_assets=[a for a in assets if a != held],
                        calibration_assets=sorted({c["asset"] for c, _ in controls}),
                        threshold=threshold,
                        threshold_audit=audit,
                        model_sha256=digest(model_path),
                        **auc_diagnostics(model, local, held, family),
                    )
                )
                for c in local:
                    if c["asset"] == held:
                        score = predict(model, c, family)
                        rows.append(dict(family=family, **evaluate(c, score, threshold)))
                        pd.DataFrame(dict(time=c["times"], score=score, valid=c["valid"])).to_csv(
                            args.models
                            / f"{family}_{farm.replace(' ', '_')}_{c['event_id']}_scores.csv",
                            index=False,
                        )
                (args.output / "events.json").write_text(json.dumps(rows, indent=2))
                (args.output / "folds.json").write_text(json.dumps(folds, indent=2))
                (args.output / "warnings.json").write_text(json.dumps(fit_warnings, indent=2))
                print(f"{family} / {farm} / turbine {held}: completed", flush=True)
        timings.append(dict(family=family, elapsed_seconds=time.monotonic() - began))
        (args.output / "timings.json").write_text(json.dumps(timings, indent=2))
        (args.output / "summary.json").write_text(
            json.dumps(summarize(pd.DataFrame(rows)), indent=2)
        )
    pd.DataFrame(rows).to_csv(args.output / "events.csv", index=False)
    print(json.dumps(summarize(pd.DataFrame(rows)), indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
