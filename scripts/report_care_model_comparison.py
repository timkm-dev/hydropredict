"""Mentor-facing aggregate evidence for the completed CARE family comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("docs/results/care-model-comparison-v4")
    events = pd.read_csv(root / "events.csv")
    folds = pd.DataFrame(json.loads((root / "folds.json").read_text()))
    assert len(events) == 148 and len(folds) == 56
    for row in folds.itertuples():
        assert row.held_asset not in row.training_assets
        assert row.held_asset not in row.calibration_assets
    previous = pd.read_csv("docs/results/care-supervised-v3/events.csv")
    for _, data in events.groupby("family"):
        merged = data.merge(
            previous, on=["farm", "event_id"], suffixes=("_new", "_old"), validate="one_to_one"
        )
        assert len(merged) == 37
        assert (merged.source_sha256_new == merged.source_sha256_old).all()
        assert (merged.prediction_rows_new == merged.prediction_rows_old).all()
        np.testing.assert_allclose(merged.valid_fraction_new, merged.valid_fraction_old)
    rows = []
    for folder, name in [
        ("care-baseline-v1", "Isolation Forest"),
        ("care-residual-v2", "Extra Trees residuals"),
        ("care-supervised-v3", "Gradient boosting"),
    ]:
        for r in json.loads((Path("docs/results") / folder / "summary.json").read_text()):
            rows.append(dict(model=name, **r))
    names = {
        "logistic": "Logistic regression",
        "gaussian_nb": "Gaussian naive Bayes",
        "random_forest": "Random forest",
        "rbf_svm_approx": "Approx. RBF SVM",
    }
    for r in json.loads((root / "summary.json").read_text()):
        rows.append(dict(model=names[r["family"]], **r))
    combined = pd.DataFrame(rows)
    combined[
        [
            "model",
            "farm",
            "anomaly_files",
            "detected_24h",
            "normal_episodes_per_30_days",
            "normal_alert_fraction",
            "gate_pass",
        ]
    ].to_csv(root / "all-models.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, farm in zip(axes, ["Wind Farm A", "Wind Farm B"], strict=True):
        data = combined[combined.farm.eq(farm)]
        ax.scatter(
            data.normal_episodes_per_30_days,
            100 * data.detected_24h / data.anomaly_files,
            s=55,
            color="#147d92",
        )
        offsets = [(5, 5), (5, 5), (5, -12), (5, 8), (5, -15), (5, 18), (5, -25)]
        for r, offset in zip(data.itertuples(), offsets, strict=True):
            ax.annotate(
                r.model,
                (r.normal_episodes_per_30_days, 100 * r.detected_24h / r.anomaly_files),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
        ax.axvline(1, color="#ba583d", linestyle="--")
        ax.axhline(50, color="#ba583d", linestyle="--")
        ax.set_title(farm)
        ax.set_xlabel("Normal alarm episodes per 30 observed days")
        ax.set_ylabel("Early anomaly cases detected (%)")
        ax.set_ylim(-8, 110)
        ax.set_xlim(-0.1, max(6, float(data.normal_episodes_per_30_days.max()) * 1.35))
    fig.suptitle(
        "CARE development screen: detection versus normal-alarm burden\n"
        "Desired region: above 50%, left of 1; normal-row alarm limit also applies",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(root / "model-comparison.png", dpi=160)
    folds[["family", "farm", "held_asset", "threshold", "fitting_auc", "held_out_auc"]].to_csv(
        root / "fold-diagnostics.csv", index=False
    )
    events.groupby(["family", "farm", "asset", "label"]).agg(
        files=("event_id", "size"),
        detected_24h=("detected_24h", "sum"),
        episodes=("episodes", "sum"),
        observed_days=("observed_days", "sum"),
        alert_rows=("alert_rows", "sum"),
    ).to_csv(root / "turbines.csv")
    print(
        combined[
            [
                "model",
                "farm",
                "detected_24h",
                "anomaly_files",
                "normal_episodes_per_30_days",
                "normal_alert_fraction",
                "gate_pass",
            ]
        ].to_string(index=False)
    )
    print("Warnings:", json.loads((root / "warnings.json").read_text()))
    print("Timings:", json.loads((root / "timings.json").read_text()))
    print(
        "Median AUCs:",
        folds.groupby(["family", "farm"])[["fitting_auc", "held_out_auc"]].median().to_string(),
    )


if __name__ == "__main__":
    main()
