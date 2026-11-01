"""Fixed EIF/IF and conditional-vibration development experiment."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

from scripts.eif_reference import ExtendedForest
from scripts.supervised_precursor import (
    CHANNELS,
    dump,
    labels,
    metrics,
    persistent,
    read,
    sha,
    split_rows,
)

FOLDS = {
    "primary": ["2018-08-01", "2019-01-01", "2019-02-01", "2019-05-01"],
    "november": ["2018-08-01", "2018-10-01", "2018-11-01", "2018-12-01"],
    "december": ["2018-08-01", "2018-11-01", "2018-12-01", "2019-01-01"],
}
REGRESSION = dict(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    max_depth=4,
    min_samples_leaf=30,
    l2_regularization=1,
    early_stopping=False,
    random_state=42,
)


class Candidate:
    def __init__(self, name):
        self.name = name

    def fit(self, frame):
        self.scaler = StandardScaler().fit(frame[CHANNELS])
        values = self.scaler.transform(frame[CHANNELS])
        if self.name == "eif":
            self.model = ExtendedForest().fit(values)
        elif self.name == "if":
            self.model = IsolationForest(
                n_estimators=500, max_samples=2048, contamination="auto", random_state=42, n_jobs=1
            ).fit(values)
        else:
            self.model = [
                HistGradientBoostingRegressor(**REGRESSION).fit(frame[["V5", "V6"]], frame[c])
                for c in CHANNELS[:4]
            ]
        return self

    def score(self, frame):
        if self.name == "eif":
            return self.model.score(self.scaler.transform(frame[CHANNELS]))
        if self.name == "if":
            return -self.model.score_samples(self.scaler.transform(frame[CHANNELS]))
        predictions = np.column_stack([m.predict(frame[["V5", "V6"]]) for m in self.model])
        return (np.abs(frame[CHANNELS[:4]].to_numpy() - predictions) / self.scaler.scale_[:4]).max(
            axis=1
        )


def primary_pass(result):
    return (
        set(result["matched"]["warned_groups"]) > set(result["lstm_matched"]["warned_groups"])
        and result["native"]["nominal_alert_fraction"] <= 0.02
    )


def stable(results, name):
    return all(
        r["native"]["warned_groups"] and r["native"]["nominal_alert_fraction"] <= 0.02
        for r in results
        if r["candidate"] == name and r["fold"] != "primary"
    )


def evaluate(name, fold, timeline, faults, normal, baseline, output):
    bounds = pd.to_datetime(FOLDS[fold], utc=True)
    frames = [
        split_rows(timeline, faults, a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    fit, calibration, evaluation = frames
    fitting = fit[fit.t.isin(normal) & (labels(fit.t, faults) == 0)]
    cal_nominal = calibration.t.isin(normal) & (labels(calibration.t, faults) == 0)
    model = Candidate(name).fit(fitting)
    cal_scores = model.score(calibration)
    threshold = float(np.quantile(cal_scores[cal_nominal], 0.98, method="higher"))
    identity = f"{fold}-{name}"
    dump(
        output / f"{identity}-frozen.json",
        dict(
            candidate=name,
            fold=fold,
            boundaries=[b.isoformat() for b in bounds],
            threshold=threshold,
            fitting_rows=len(fitting),
            calibration_rows=len(calibration),
            calibration_nominal_rows=int(cal_nominal.sum()),
            persistence=3,
            seed=42,
            ntrees=500 if name != "conditional" else None,
            sample_size=2048 if name != "conditional" else None,
            regression=REGRESSION if name == "conditional" else None,
        ),
    )
    joblib.dump(model, output / f"{identity}.joblib")
    score = model.score(evaluation)
    replay = joblib.load(output / f"{identity}.joblib").score(evaluation)
    np.testing.assert_allclose(score, replay, rtol=0, atol=0)
    np.testing.assert_allclose(
        score[:19], model.score(evaluation.iloc[:19]), rtol=1e-12, atol=1e-12
    )
    scores = pd.DataFrame({"t": evaluation.t, "score": score})
    scores["point_exceedance"] = scores.score.gt(threshold)
    scores["is_alert"] = persistent(scores.t, scores.point_exceedance)
    scores.to_csv(output / f"{identity}-scores.csv", index=False)
    cal = pd.DataFrame({"t": calibration.t, "score": cal_scores})
    cal["point_exceedance"] = cal.score.gt(threshold)
    cal["is_alert"] = persistent(cal.t, cal.point_exceedance)
    cal.to_csv(output / f"{identity}-calibration.csv", index=False)
    if name == "conditional":
        contributions = evaluation[["t", *CHANNELS]].copy()
        for i, channel in enumerate(CHANNELS[:4]):
            contributions[f"{channel}_expected"] = model.model[i].predict(evaluation[["V5", "V6"]])
            contributions[f"{channel}_residual_sd"] = (
                evaluation[channel] - contributions[f"{channel}_expected"]
            ).abs() / model.scaler.scale_[i]
        contributions.to_csv(output / f"{identity}-residuals.csv", index=False)
    eval_faults = [f for f in faults if bounds[2] <= f < bounds[3]]
    nominal = normal[labels(normal, faults) == 0]
    point = scores.copy()
    point["is_alert"] = point.point_exceedance
    result = dict(
        candidate=name,
        fold=fold,
        native=metrics(scores, eval_faults, nominal),
        point_only_diagnostic=metrics(point, eval_faults, nominal),
        threshold=threshold,
        calibration_nominal_fraction=float(cal.loc[cal_nominal, "is_alert"].mean()),
        calibration_point_fraction=float(cal.loc[cal_nominal, "point_exceedance"].mean()),
        serialized_replay_exact=True,
        prefix_score_invariant=True,
    )
    if fold == "primary":
        common = scores.t[scores.t.isin(baseline.t)]
        result["matched"] = metrics(scores[scores.t.isin(common)], eval_faults, nominal)
        result["lstm_matched"] = metrics(baseline[baseline.t.isin(common)], eval_faults, nominal)
        result["lstm_native"] = metrics(baseline, eval_faults, nominal)
        result["primary_improvement"] = primary_pass(result)
    dump(output / f"{identity}-result.json", result)
    print(
        json.dumps(
            dict(
                candidate=name,
                fold=fold,
                warned=result["native"]["warned_groups"],
                eligible=result["native"]["eligible_groups"],
                nominal_fraction=result["native"]["nominal_alert_fraction"],
            )
        ),
        flush=True,
    )
    return result


def run(package, diagnostics, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = json.loads((package / "checksums.json").read_text())
    names = [f"{part}-observations.csv" for part in ("fit", "calibration", "development")]
    for name in [*names, "faults.csv", "provenance.csv"]:
        if sha(package / name) != checks[name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    timeline = pd.concat([read(package / name) for name in names], ignore_index=True)
    if timeline.t.duplicated().any() or not timeline.t.is_monotonic_increasing:
        raise ValueError("Invalid timeline")
    if timeline.t.ge(pd.Timestamp("2019-05-01", tz="UTC")).any():
        raise ValueError("Later measurements forbidden")
    faults = read(package / "faults.csv").t.tolist()
    provenance = read(package / "provenance.csv")
    normal = provenance.t[
        provenance.source_records.map(
            lambda s: any(r["source_id"] == "normal" for r in json.loads(s))
        )
    ]
    expected = json.loads((diagnostics / "checksums.json").read_text())["lstm-scores.csv"]
    if sha(diagnostics / "lstm-scores.csv").upper() != expected.upper():
        raise ValueError("Baseline checksum mismatch")
    baseline = read(diagnostics / "lstm-scores.csv")
    dump(
        output / "study.json",
        dict(
            protocol_sha256=sha("docs/eif-context-protocol.md"),
            script_sha256=sha(__file__),
            eif_sha256=sha("scripts/eif_reference.py"),
            shared_script_sha256=sha("scripts/supervised_precursor.py"),
            package_manifest_sha256=sha(package / "checksums.json"),
            baseline_sha256=expected,
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
            development_only=True,
        ),
    )
    results = []
    for name in ("if", "eif"):
        for fold in FOLDS:
            results.append(evaluate(name, fold, timeline, faults, normal, baseline, output))
            dump(output / "results.json", results)
    passes = [
        r["candidate"]
        for r in results
        if r["fold"] == "primary" and r["primary_improvement"] and stable(results, r["candidate"])
    ]
    conditional_triggered = not passes
    if conditional_triggered:
        for fold in FOLDS:
            results.append(
                evaluate("conditional", fold, timeline, faults, normal, baseline, output)
            )
            dump(output / "results.json", results)
        primary = next(
            r for r in results if r["candidate"] == "conditional" and r["fold"] == "primary"
        )
        if primary["primary_improvement"] and stable(results, "conditional"):
            passes.append("conditional")
    dump(
        output / "decision.json",
        dict(
            candidates_passing_both_gates=passes,
            conditional_triggered=conditional_triggered,
            retain_baseline=not passes,
            independent_confirmation=False,
        ),
    )
    dump(output / "checksums.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.package, args.diagnostics, args.output)
