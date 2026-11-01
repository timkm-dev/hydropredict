"""Fixed inner-normal quantile calibration on saved outer CARE classifiers."""

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.compare_care_models import FAMILIES, evaluate, fast_alerts, fit, summarize
from scripts.train_care_baseline import digest


def native_scores(model, x, family):
    if family in ["logistic", "rbf_svm_approx"]:
        return model.decision_function(x)
    if family == "gaussian_nb":
        logp = model.predict_log_proba(x)
        return logp[:, 1] - logp[:, 0]
    return model.predict_proba(x)[:, 1]


def predict_native(model, case, family):
    scores = np.full(len(case["x"]), np.nan)
    values = native_scores(model, case["x"].loc[case["valid"]], family)
    assert np.isfinite(values).all()
    scores[case["valid"]] = values
    return scores


def calibrate_quantiles(controls):
    values = np.concatenate([scores[np.isfinite(scores)] for _, scores in controls])
    assert len(values) > 0
    audit = []
    for q in [0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 1.0]:
        threshold = float(np.quantile(values, q))
        rows = alert_rows = episodes = 0
        for case, score in controls:
            assert case["label"] == "normal"
            alerts, starts = fast_alerts(case["times"], score > threshold)
            rows += len(score)
            alert_rows += int(alerts.sum())
            episodes += int(starts.sum())
        rate = episodes / (rows / 144) * 30
        fraction = alert_rows / rows
        audit.append(
            dict(
                quantile=q, threshold=threshold, episodes_per_30_days=rate, alert_fraction=fraction
            )
        )
        if rate <= 1 and fraction <= 0.01:
            return threshold, q, audit
    raise AssertionError("Max-normal calibration endpoint must pass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.scores.mkdir(parents=True, exist_ok=False)
    previous = Path("docs/results/care-model-comparison-v4")
    old_config = json.loads((previous / "config.json").read_text())
    cache = Path(".local/care-supervised-v3-diagnostic-features.joblib")
    assert digest(cache) == old_config["cache_sha256"]
    assert digest(Path("scripts/compare_care_models.py")) == old_config["code_sha256"]
    cases = joblib.load(cache)
    prior_folds = json.loads((previous / "folds.json").read_text())
    assert len(prior_folds) == 56
    (args.output / "config.json").write_text(
        json.dumps(
            dict(
                protocol_sha256=digest(Path("docs/care-model-calibration-addendum.md")),
                code_sha256=digest(Path(__file__)),
                cache_sha256=digest(cache),
                previous_config_sha256=digest(previous / "config.json"),
            ),
            indent=2,
        )
    )
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
                        pairs[(left, right, c["event_id"])] = predict_native(model, c, family)
            for held in assets:
                controls = []
                for c in local:
                    if c["asset"] != held and c["label"] == "normal":
                        left, right = sorted([held, c["asset"]])
                        controls.append((c, pairs[(left, right, c["event_id"])]))
                threshold, q, audit = calibrate_quantiles(controls)
                model_path = (
                    Path(".local/care-model-comparison-v4-models")
                    / f"{family}_{farm.replace(' ', '_')}_{held}.joblib"
                )
                old_fold = next(
                    f
                    for f in prior_folds
                    if f["family"] == family and f["farm"] == farm and f["held_asset"] == held
                )
                assert digest(model_path) == old_fold["model_sha256"]
                model = joblib.load(model_path)["model"]
                folds.append(
                    dict(
                        family=family,
                        farm=farm,
                        held_asset=held,
                        threshold=threshold,
                        quantile=q,
                        threshold_audit=audit,
                        model_sha256=digest(model_path),
                        training_assets=old_fold["training_assets"],
                        calibration_assets=sorted({c["asset"] for c, _ in controls}),
                    )
                )
                for c in local:
                    if c["asset"] == held:
                        scores = predict_native(model, c, family)
                        rows.append(dict(family=family, **evaluate(c, scores, threshold)))
                        pd.DataFrame(dict(time=c["times"], score=scores, valid=c["valid"])).to_csv(
                            args.scores
                            / f"{family}_{farm.replace(' ', '_')}_{c['event_id']}_scores.csv",
                            index=False,
                        )
                (args.output / "events.json").write_text(json.dumps(rows, indent=2))
                (args.output / "folds.json").write_text(json.dumps(folds, indent=2))
                (args.output / "warnings.json").write_text(json.dumps(fit_warnings, indent=2))
                print(f"Native quantiles: {family} / {farm} / {held} completed", flush=True)
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
