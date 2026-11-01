"""Run the two fixed CWRU evaluations and export the primary bundle (#9)."""

import json
import platform
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

from hydropredict.models.bearing import (
    CLASSES,
    FEATURES,
    PARAMS,
    BearingService,
    extract,
    read_signal,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(model, entries, cache):
    actual, predicted, outputs, wy, wp = [], [], [], [], []
    for row in entries.itertuples():
        features = cache[row.recording]
        probabilities = model.predict_proba(features)
        prediction = int(probabilities.mean(axis=0).argmax())
        actual.append(CLASSES.index(row.label))
        predicted.append(prediction)
        wy.extend([CLASSES.index(row.label)] * len(features))
        wp.extend(probabilities.argmax(axis=1).tolist())
        outputs.append(
            dict(
                recording=row.recording,
                actual=row.label,
                predicted=CLASSES[prediction],
                windows=len(features),
            )
        )
    return dict(
        recordings=len(actual),
        correct=int(np.sum(np.array(actual) == predicted)),
        accuracy=accuracy_score(actual, predicted),
        macro_f1=f1_score(actual, predicted, average="macro"),
        confusion_matrix=confusion_matrix(actual, predicted, labels=range(4)).tolist(),
        per_class=classification_report(
            actual,
            predicted,
            labels=range(4),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        window_accuracy=accuracy_score(wy, wp),
        predictions=outputs,
    )


def run():
    root = Path("models/cwru-xgb-v1")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "model.ubj").exists():
        raise ValueError("A completed model already exists; use a new output directory.")
    report = Path("docs/results/cwru-v1")
    report.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv("data/cwru_manifest.csv")
    cache = {}
    accounting = {}
    for row in manifest.itertuples():
        content = Path(row.local_filename).read_bytes()
        assert sha256(content).hexdigest() == row.sha256
        cache[row.recording], accounting[row.recording] = extract(
            read_signal(content, row.recording)
        )
    masks = {
        "primary": (manifest.load_hp.isin([0, 1]), manifest.load_hp.eq(3)),
        "size_challenge": (
            ~manifest.diameter_mils.eq(14)
            & ~((manifest.label == "Healthy") & (manifest.load_hp == 3)),
            manifest.diameter_mils.eq(14)
            | ((manifest.label == "Healthy") & (manifest.load_hp == 3)),
        ),
    }
    results = {}
    for name, (train_mask, test_mask) in masks.items():
        train, test = manifest[train_mask], manifest[test_mask]
        assert not set(train.recording) & set(test.recording)
        x = np.concatenate([cache[r] for r in train.recording])
        y = np.concatenate(
            [np.full(len(cache[r.recording]), CLASSES.index(r.label)) for r in train.itertuples()]
        )
        weights = np.concatenate(
            [
                np.full(
                    len(cache[r.recording]),
                    1 / (len(cache[r.recording]) * sum(train.label == r.label)),
                )
                for r in train.itertuples()
            ]
        )
        weights *= len(weights) / weights.sum()
        model = XGBClassifier(**PARAMS).fit(x, y, sample_weight=weights)
        destination = root if name == "primary" else root / name
        destination.mkdir(exist_ok=True)
        model.save_model(destination / "model.ubj")
        results[name] = dict(
            train_recordings=train.recording.tolist(), test=evaluate(model, test, cache)
        )
        if name == "primary":
            results[name]["validation"] = evaluate(model, manifest[manifest.load_hp.eq(2)], cache)
        print(
            name,
            json.dumps(
                {
                    k: v
                    for k, v in results[name]["test"].items()
                    if k not in ["per_class", "predictions"]
                }
            ),
            flush=True,
        )
        write(
            destination / "metadata.json",
            dict(
                model_version=f"cwru-xgb-v1-{name}",
                classes=CLASSES,
                features=FEATURES,
                params=PARAMS,
                sample_rate=48000,
                analysis_rate=12000,
                window_length=4096,
                evaluation=results[name]["test"],
                training_recordings=train.recording.tolist(),
            ),
        )
        loaded = BearingService(destination)
        for row in test.itertuples():
            np.testing.assert_array_equal(
                model.predict_proba(cache[row.recording]),
                loaded.model.predict_proba(cache[row.recording]),
            )
        assert (
            loaded.assess(Path(test.iloc[0].local_filename).read_bytes(), test.iloc[0].recording)[
                "accounting"
            ]["windows"]
            > 0
        )
    write(report / "results.json", results)
    write(report / "accounting.json", accounting)
    write(root / "results.json", results)
    write(
        root / "environment.json",
        dict(python=platform.python_version(), xgboost=xgboost.__version__, numpy=np.__version__),
    )
    with zipfile.ZipFile(root / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for file in [
            "scripts/train_cwru.py",
            "scripts/acquire_cwru.py",
            "src/hydropredict/models/bearing.py",
            "docs/cwru-protocol.md",
            "data/cwru_manifest.csv",
        ]:
            z.write(file, file)
    write(
        root / "checksums.json",
        {
            str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )


if __name__ == "__main__":
    run()
