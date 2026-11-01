"""Verify and visualize the nested grouped CARE experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("docs/results/care-supervised-v3")
    events = pd.read_csv(root / "events.csv")
    folds = json.loads((root / "folds.json").read_text())
    assert len(events) == 37 and len(folds) == 14
    for fold in folds:
        assert fold["held_asset"] not in fold["training_assets"]
        assert fold["held_asset"] not in fold["calibration_assets"]
    old = pd.read_csv("docs/results/care-residual-v2/events.csv")
    joined = events.merge(
        old, on=["farm", "event_id"], suffixes=("_v3", "_v2"), validate="one_to_one"
    )
    assert (joined.source_sha256_v3 == joined.source_sha256_v2).all()
    assert (joined.prediction_rows_v3 == joined.prediction_rows_v2).all()
    summaries = [
        json.loads((Path("docs/results") / version / "summary.json").read_text())
        for version in ["care-baseline-v1", "care-residual-v2", "care-supervised-v3"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for j, (key, title, gate) in enumerate(
        zip(
            ["detected_24h", "normal_episodes_per_30_days", "normal_alert_fraction"],
            [
                "Early event detections (%)",
                "Normal episodes / 30 observed days",
                "Normal rows alerted (%)",
            ],
            [50, 1, 1],
            strict=True,
        )
    ):
        ax = axes[j]
        for k, summary in enumerate(summaries):
            values = [s[key] for s in summary]
            if j == 0:
                values = [
                    100 * v / s["anomaly_files"] for v, s in zip(values, summary, strict=True)
                ]
            if j == 2:
                values = [100 * v for v in values]
            ax.bar(
                np.arange(2) + (k - 1) * 0.25,
                values,
                width=0.25,
                label=["Isolation Forest", "Conditional residuals", "Grouped supervised"][k],
                color=["#9aa6ac", "#58a3b1", "#164b6d"][k],
            )
        ax.set_xticks([0, 1], ["Farm A", "Farm B"])
        ax.axhline(gate, color="#bd5837", linestyle="--", label="Screening gate")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle("CARE development evidence; early = >=24h before annotated interval end")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    print("Thresholds:", [(f["farm"], f["held_asset"], f["threshold"]) for f in folds])
    print(
        events[events.label.eq("anomaly")][
            [
                "farm",
                "event_id",
                "asset",
                "detected_24h",
                "end_relative_lead_hours",
                "preexisting_alert",
            ]
        ].to_string(index=False)
    )
    events.groupby(["farm", "asset", "label"]).agg(
        files=("event_id", "size"),
        detected_24h=("detected_24h", "sum"),
        episodes=("episodes", "sum"),
        observed_days=("observed_days", "sum"),
        alert_rows=("alert_rows", "sum"),
    ).to_csv(root / "turbines.csv")


if __name__ == "__main__":
    main()
