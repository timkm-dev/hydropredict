"""Bounded earlier-only transfer study; never loads development measurements."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.supervised_precursor import (
    CHANNELS,
    PARAMS,
    dump,
    features,
    labels,
    metrics,
    persistent,
    read,
    sha,
    split_rows,
)

CANDIDATES = ("absolute_hgb", "absolute_logistic", "relative_hgb", "relative_logistic")


def relative_features(frame):
    dt = frame.t.diff().dt.total_seconds()
    parts = []
    for _, group in frame.groupby((~dt.between(150, 450)).cumsum()):
        values = group.set_index("t")[CHANNELS].astype(float)
        x = pd.Series((values.index - values.index[0]).total_seconds() / 3600, index=values.index)
        rolling = values.rolling("30min", min_periods=3)
        xr = x.rolling("30min", min_periods=3)
        denominator = (x * x).rolling("30min", min_periods=3).mean() - xr.mean() ** 2
        out = pd.DataFrame(index=values.index)
        for channel in CHANNELS:
            numerator = (values[channel] * x).rolling(
                "30min", min_periods=3
            ).mean() - rolling.mean()[channel] * xr.mean()
            out[f"{channel}_deviation"] = values[channel] - rolling.mean()[channel]
            out[f"{channel}_std"] = rolling.std()[channel]
            out[f"{channel}_slope"] = numerator / denominator
        out = out[out.index >= out.index[0] + pd.Timedelta(minutes=30)].dropna()
        out.insert(0, "t", out.index)
        parts.append(out.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def model_for(name):
    if name.endswith("hgb"):
        return HistGradientBoostingClassifier(**PARAMS)
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    )


def choose(results):
    admissible = []
    for name in CANDIDATES:
        folds = [r for r in results if r["candidate"] == name]
        if len(folds) != 2 or not all(
            r["metrics"]["warned_groups"]
            and r["metrics"]["nominal_alert_fraction"] is not None
            and r["metrics"]["nominal_alert_fraction"] <= 0.02
            for r in folds
        ):
            continue
        warned = sum(len(r["metrics"]["warned_groups"]) for r in folds)
        alerts = sum(r["metrics"]["nominal_alerts"] for r in folds)
        total = sum(r["metrics"]["nominal_points"] for r in folds)
        admissible.append((-warned, alerts / total, name))
    return min(admissible)[2] if admissible else None


def prepare(timeline, faults, normal, start, end):
    raw = split_rows(timeline, faults, start, end)
    absolute = features(raw)
    relative = relative_features(raw)
    common = absolute.t[absolute.t.isin(relative.t)]
    absolute = absolute[absolute.t.isin(common)].reset_index(drop=True)
    relative = relative[relative.t.isin(common)].reset_index(drop=True)
    if not absolute.t.equals(relative.t):
        raise AssertionError("Feature endpoint alignment failed")
    target = labels(absolute.t, faults)
    nominal = absolute.t.isin(normal) & (target == 0)
    return dict(
        absolute=absolute,
        relative=relative,
        target=target,
        nominal=nominal,
        observations=len(raw),
        excluded_warmup=len(raw) - len(absolute),
    )


def evaluate(name, partitions, faults, output, fold):
    mode = name.split("_")[0]
    fit, cal, val = partitions
    columns = [c for c in fit[mode] if c != "t"]
    model = model_for(name)
    model.fit(fit[mode][columns], fit["target"])
    cal_score = model.predict_proba(cal[mode][columns])[:, 1]
    threshold = float(np.quantile(cal_score[cal["nominal"]], 0.98, method="higher"))
    identity = f"{fold}-{name}"
    # Persist the decision before validation scoring.
    dump(
        output / f"{identity}-frozen.json",
        dict(
            candidate=name,
            threshold=threshold,
            predictors=columns,
            model_parameters=str(model.get_params()),
            persistence=3,
        ),
    )
    joblib.dump(model, output / f"{identity}.joblib")
    scores = pd.DataFrame(
        {"t": val[mode].t, "score": model.predict_proba(val[mode][columns])[:, 1]}
    )
    replay = joblib.load(output / f"{identity}.joblib").predict_proba(val[mode][columns])[:, 1]
    if not np.array_equal(scores.score.to_numpy(), replay):
        raise AssertionError("Replay mismatch")
    scores["is_alert"] = persistent(scores.t, scores.score.gt(threshold))
    scores["target"] = val["target"]
    scores.to_csv(output / f"{identity}-scores.csv", index=False)
    return dict(
        candidate=name,
        fold=fold,
        threshold=threshold,
        replay_exact=True,
        counts=[
            dict(
                observations=p["observations"],
                endpoints=len(p[mode]),
                positive=int(p["target"].sum()),
                warmup_excluded=p["excluded_warmup"],
            )
            for p in partitions
        ],
        metrics=metrics(scores, faults, val[mode].t[val["nominal"]]),
    )


def run(package, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = json.loads((package / "checksums.json").read_text())
    # Read only these pre-February observations. Hashing is also restricted to those inputs.
    for name in (
        "fit-observations.csv",
        "calibration-observations.csv",
        "faults.csv",
        "provenance.csv",
    ):
        if sha(package / name) != checks[name]:
            raise ValueError(f"Changed input: {name}")
    timeline = pd.concat(
        [read(package / f"{part}-observations.csv") for part in ("fit", "calibration")],
        ignore_index=True,
    )
    cutoff = pd.Timestamp("2019-02-01", tz="UTC")
    if timeline.t.ge(cutoff).any():
        raise ValueError("Development measurements are forbidden")
    faults = read(package / "faults.csv").t
    faults = faults[faults.lt(cutoff)].tolist()
    provenance = read(package / "provenance.csv")
    provenance = provenance[provenance.t.lt(cutoff)]
    normal = provenance.t[
        provenance.source_records.map(
            lambda s: any(r["source_id"] == "normal" for r in json.loads(s))
        )
    ]
    dump(
        output / "study.json",
        dict(
            candidates=CANDIDATES,
            script_sha256=sha(__file__),
            protocol_sha256=sha("docs/earlier-backtest-protocol.md"),
            shared_script_sha256=sha("scripts/supervised_precursor.py"),
            package_manifest_sha256=sha(package / "checksums.json"),
            python=platform.python_version(),
            sklearn=sklearn.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            development_measurements_loaded=False,
        ),
    )
    results = []
    for fold, dates in (
        ("november", ["2018-08-01", "2018-10-01", "2018-11-01", "2018-12-01"]),
        ("december", ["2018-08-01", "2018-11-01", "2018-12-01", "2019-01-01"]),
    ):
        bounds = pd.to_datetime(dates, utc=True)
        partitions = [
            prepare(timeline, faults, normal, a, b)
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        eval_faults = [f for f in faults if bounds[2] <= f < bounds[3]]
        for name in CANDIDATES:
            result = evaluate(name, partitions, eval_faults, output, fold)
            results.append(result)
            dump(output / "selection-results.json", results)
            print(
                json.dumps(
                    dict(
                        candidate=name,
                        fold=fold,
                        warned=result["metrics"]["warned_groups"],
                        eligible=result["metrics"]["eligible_groups"],
                        nominal_fraction=result["metrics"]["nominal_alert_fraction"],
                    )
                ),
                flush=True,
            )
    candidate = choose(results)
    decision = dict(selected=candidate, confirmation=None, promising=False)
    dump(output / "decision.json", decision)
    if candidate is not None:
        bounds = pd.to_datetime(["2018-08-01", "2018-12-01", "2019-01-01", "2019-02-01"], utc=True)
        partitions = [
            prepare(timeline, faults, normal, a, b)
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        result = evaluate(
            candidate,
            partitions,
            [f for f in faults if bounds[2] <= f < bounds[3]],
            output,
            "january",
        )
        decision["confirmation"] = result
        decision["promising"] = bool(
            result["metrics"]["warned_groups"]
            and result["metrics"]["nominal_alert_fraction"] <= 0.02
        )
        dump(output / "decision.json", decision)
    dump(output / "checksums.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.package, args.output)
