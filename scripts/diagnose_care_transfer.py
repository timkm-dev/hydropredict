"""Post-fit descriptive fit-versus-held-turbine discrimination; no tuning."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from scripts.research_care_supervised import load_cases
from scripts.train_care_baseline import digest


def main():
    cache = Path(".local/care-supervised-v3-diagnostic-features.joblib")
    manifest = cache.with_suffix(".json")
    identity = dict(
        upstream=digest(Path("docs/results/care-residual-v2/events.json")),
        feature_code=digest(Path("scripts/research_care_supervised.py")),
    )
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == identity:
        cases = joblib.load(cache)
    else:
        cases = load_cases(
            Path(r"C:\Users\tmuny\Downloads\CARE_To_Compare\CARE_To_Compare"),
            Path(".local/care-residual-v2-models"),
        )
        joblib.dump(cases, cache, compress=3)
        manifest.write_text(json.dumps(identity))
    root = Path("docs/results/care-supervised-v3")
    folds = json.loads((root / "folds.json").read_text())
    rows = []
    for fold in folds:
        model = joblib.load(
            Path(".local/care-supervised-v3-models")
            / f"{fold['farm'].replace(' ', '_')}_{fold['held_asset']}.joblib"
        )["model"]
        result = dict(farm=fold["farm"], held_asset=fold["held_asset"])
        for stage in ["fitting", "held_out"]:
            local = [
                c
                for c in cases
                if c["farm"] == fold["farm"]
                and (c["asset"] == fold["held_asset"]) == (stage == "held_out")
                and len(c["fit_idx"])
            ]
            labels = []
            predictions = []
            weights = []
            for c in local:
                x = c["x"].iloc[c["fit_idx"]]
                labels.extend([int(c["label"] == "anomaly")] * len(x))
                predictions.extend(model.predict_proba(x)[:, 1])
                weights.extend([1 / len(x)] * len(x))
            result[stage + "_file_weighted_row_auc"] = (
                float(roc_auc_score(labels, predictions, sample_weight=weights))
                if len(set(labels)) == 2
                else None
            )
        rows.append(result)
    pd.DataFrame(rows).to_csv(root / "fit-versus-transfer.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
