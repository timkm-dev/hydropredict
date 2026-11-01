"""Bounded retrospective sequence-pattern comparison for issue #20.

No final holdout claim: these development periods have already been inspected.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from hydropredict.data.sequences import build_sequences
from scripts.compare_eif_context import FOLDS
from scripts.supervised_precursor import (
    dump,
    groups,
    labels,
    metrics,
    persistent,
    read,
    sha,
    split_rows,
)


class PatternModel:
    def __init__(self, family, representation):
        self.family = family
        self.representation = representation

    def transform(self, values):
        x = (values - self.mean) / self.scale
        if self.representation == "shape":
            x = x - x.mean(axis=1, keepdims=True)
            x = x / np.maximum(x.std(axis=1, keepdims=True), 0.05)
        return x.astype(np.float32)

    def convolution(self, x):
        features = []
        for indices, weights, bias in self.kernels:
            # Each sequence ends at the prediction time; no future samples exist.
            windows = x[:, indices, :]
            response = np.einsum("ntkc,mkc->ntm", windows, weights, optimize=True) + bias
            features.extend([response.max(axis=1), (response > 0).mean(axis=1)])
        return np.concatenate(features, axis=1)

    def fit(self, values, y, event_ids):
        self.mean = values.mean(axis=(0, 1), keepdims=True)
        self.scale = np.maximum(values.std(axis=(0, 1), keepdims=True), 1e-6)
        x = self.transform(values)
        rng = np.random.default_rng(42)
        if self.family == "dictionary":
            # Cap each fault group's contribution; overlapping rows cannot dominate the bank.
            positive = []
            for group in np.unique(event_ids[y == 1]):
                ids = np.flatnonzero((y == 1) & (event_ids == group))
                positive.extend(ids[np.linspace(0, len(ids) - 1, min(24, len(ids))).astype(int)])
            negative = np.flatnonzero(y == 0)
            negative = rng.choice(negative, min(2048, len(negative)), replace=False)
            self.positive = x[positive].reshape(len(positive), -1)
            self.negative = x[negative].reshape(len(negative), -1)
        else:
            self.kernels = []
            for length, dilation in [(3, 1), (3, 2), (5, 1), (7, 1)]:
                starts = np.arange(x.shape[1] - (length - 1) * dilation)
                indices = starts[:, None] + np.arange(length)[None, :] * dilation
                weights = rng.normal(size=(64, length, 6)).astype(np.float32)
                weights -= weights.mean(axis=1, keepdims=True)
                weights /= np.sqrt(length * 6)
                self.kernels.append((indices, weights, rng.uniform(-1, 1, 64)))
            features = self.convolution(x)
            self.feature_scaler = StandardScaler().fit(features)
            weights = np.ones(len(y), dtype=float)
            for group in np.unique(event_ids[y == 1]):
                mask = (y == 1) & (event_ids == group)
                weights[mask] = 1 / mask.sum()
            weights[y == 1] *= (y == 0).sum() / weights[y == 1].sum()
            self.classifier = RidgeClassifier(alpha=10).fit(
                self.feature_scaler.transform(features), y, sample_weight=weights
            )
        return self

    def score(self, values):
        x = self.transform(values)
        if self.family == "dictionary":
            flat = x.reshape(len(x), -1)
            result = []
            for start in range(0, len(x), 512):
                chunk = flat[start : start + 512]
                pos = cdist(chunk, self.positive, "sqeuclidean").min(axis=1)
                neg = cdist(chunk, self.negative, "sqeuclidean").min(axis=1)
                result.extend(np.log((neg + 1e-8) / (pos + 1e-8)))
            return np.asarray(result)
        return self.classifier.decision_function(self.feature_scaler.transform(self.convolution(x)))


def run():
    package = Path("artifacts/supervised-precursor-v1-verified")
    output = Path("models/sequence-patterns-v1-complete")
    output.mkdir(parents=True, exist_ok=False)
    aggregate = Path("docs/results/sequence-patterns-v1")
    aggregate.mkdir(parents=True, exist_ok=True)
    checks = json.loads((package / "checksums.json").read_text())
    names = [f"{p}-observations.csv" for p in ["fit", "calibration", "development"]]
    for name in [*names, "faults.csv", "provenance.csv"]:
        assert sha(package / name) == checks[name], name
    timeline = pd.concat([read(package / name) for name in names], ignore_index=True)
    faults = read(package / "faults.csv").t.tolist()
    provenance = read(package / "provenance.csv")
    normal = provenance.t[
        provenance.source_records.map(
            lambda s: any(r["source_id"] == "normal" for r in json.loads(s))
        )
    ]
    nominal = normal[labels(normal, faults) == 0]
    baseline = read("models/lstm-diagnostics-2026-10-15-v1/lstm-scores.csv")
    group_map = {f: i for i, g in enumerate(groups(faults)) for f in g}
    results = []
    protocol = dict(
        issue=20,
        seed=42,
        sequence_length=7,
        calibration_quantile=0.98,
        families=["dictionary", "random_convolution"],
        representations=["levels", "shape"],
        alert_rules=["single", "three"],
        folds=FOLDS,
        interpretation="Retrospective exploration; not independent validation",
        script_sha256=sha(__file__),
        input_checksums=checks,
    )
    dump(output / "protocol.json", protocol)
    for fold in ["november", "december", "primary"]:
        bounds = pd.to_datetime(FOLDS[fold], utc=True)
        frames = [
            split_rows(timeline, faults, a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        batches = [build_sequences(frame) for frame in frames]
        fit, cal, evaluation = batches
        y = labels(fit.endpoints.t, faults)
        fault_ns = pd.DatetimeIndex(faults).as_unit("ns").asi8
        positions = np.searchsorted(
            fault_ns, pd.DatetimeIndex(fit.endpoints.t).as_unit("ns").asi8, side="right"
        )
        ids = np.array([group_map[faults[p]] if p < len(faults) else -1 for p in positions])
        cal_nominal = cal.endpoints.t.isin(nominal)
        eval_faults = [f for f in faults if bounds[2] <= f < bounds[3]]
        for family in protocol["families"]:
            for representation in protocol["representations"]:
                name = f"{family}-{representation}"
                model = PatternModel(family, representation).fit(fit.values, y, ids)
                cal_score = model.score(cal.values)
                threshold = float(np.quantile(cal_score[cal_nominal], 0.98, method="higher"))
                score = model.score(evaluation.values)
                np.testing.assert_allclose(
                    score[:17], model.score(evaluation.values[:17]), rtol=1e-6, atol=1e-6
                )
                joblib.dump(model, output / f"{fold}-{name}.joblib")
                np.testing.assert_allclose(
                    model.score(evaluation.values[:17]),
                    joblib.load(output / f"{fold}-{name}.joblib").score(evaluation.values[:17]),
                    rtol=0,
                    atol=0,
                )
                pd.DataFrame({"t": cal.endpoints.t, "score": cal_score}).to_csv(
                    output / f"{fold}-{name}-calibration.csv", index=False
                )
                for rule in protocol["alert_rules"]:
                    scored = pd.DataFrame({"t": evaluation.endpoints.t, "score": score})
                    scored["is_alert"] = scored.score.gt(threshold)
                    if rule == "three":
                        scored["is_alert"] = persistent(scored.t, scored.is_alert)
                    result = dict(
                        fold=fold,
                        candidate=name,
                        rule=rule,
                        threshold=threshold,
                        native=metrics(scored, eval_faults, nominal),
                    )
                    if fold == "primary":
                        common = scored.t[scored.t.isin(baseline.t)]
                        result["matched"] = metrics(
                            scored[scored.t.isin(common)], eval_faults, nominal
                        )
                        result["lstm_matched"] = metrics(
                            baseline[baseline.t.isin(common)], eval_faults, nominal
                        )
                    results.append(result)
                    scored.to_csv(output / f"{fold}-{name}-{rule}-scores.csv", index=False)
                    print(
                        json.dumps(
                            dict(
                                fold=fold,
                                model=name,
                                rule=rule,
                                warned=len(result["native"]["warned_groups"]),
                                eligible=len(result["native"]["eligible_groups"]),
                                burden=result["native"]["nominal_alert_fraction"],
                            )
                        ),
                        flush=True,
                    )
                dump(output / "results.json", results)
    dump(aggregate / "results.json", results)
    dump(aggregate / "protocol.json", protocol)
    dump(output / "checksums.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
