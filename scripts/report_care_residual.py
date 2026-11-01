"""Compare the frozen CARE models without selecting new settings."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("docs/results/care-residual-v2")
    old = Path("docs/results/care-baseline-v1")
    data = pd.read_csv(root / "events.csv")
    previous = pd.read_csv(old / "events.csv")
    merged = data.merge(
        previous, on=["farm", "event_id"], suffixes=("_v2", "_v1"), validate="one_to_one"
    )
    assert len(merged) == 37
    assert (merged.source_sha256_v2 == merged.source_sha256_v1).all()
    assert (merged.prediction_rows_v2 == merged.prediction_rows_v1).all()
    for r in data.itertuples():
        assert pd.Timestamp(r.fit_end) < pd.Timestamp(r.scale_start)
        assert pd.Timestamp(r.scale_end) < pd.Timestamp(r.calibration_start)
        assert pd.Timestamp(r.calibration_end) < pd.Timestamp(r.prediction_start)
    merged[
        [
            "farm",
            "event_id",
            "label_v2",
            "detected_24h_v1",
            "detected_24h_v2",
            "episodes_v1",
            "episodes_v2",
            "alert_rows_v1",
            "alert_rows_v2",
        ]
    ].to_csv(root / "comparison.csv", index=False)
    summaries = [json.loads((p / "summary.json").read_text()) for p in [old, root]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ["detected_24h", "normal_episodes_per_30_days", "normal_alert_fraction"]
    titles = [
        "Detected >=24h before annotated end (%)",
        "Normal alarms / 30 observation-days",
        "Normal readings alerted (%)",
    ]
    for j, (ax, key, title, gate) in enumerate(zip(axes, metrics, titles, [50, 1, 1], strict=True)):
        for k, summary in enumerate(summaries):
            values = [s[key] for s in summary]
            if j == 0:
                values = [
                    v / s["anomaly_files"] * 100 for v, s in zip(values, summary, strict=True)
                ]
            if j == 2:
                values = [v * 100 for v in values]
            ax.bar(
                np.arange(2) + (k - 0.5) * 0.36,
                values,
                width=0.36,
                label=["Isolation Forest", "Conditional temperatures"][k],
                color=["#778a98", "#147d92"][k],
            )
        ax.axhline(gate, color="#b64c37", linestyle="--", label="Screening gate")
        ax.set_xticks([0, 1], ["Farm A", "Farm B"])
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle("CARE development comparison: fixed models, same prediction rows")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    print(
        data.loc[
            data.label.eq("anomaly"),
            [
                "farm",
                "event_id",
                "detected_24h",
                "end_relative_lead_hours",
                "first_alert_status",
                "preexisting_alert",
            ],
        ].to_string(index=False)
    )
    print("Minimum validity", data.valid_fraction.min())
    print("Seconds", json.loads((root / "config.json").read_text())["elapsed_seconds"])


if __name__ == "__main__":
    main()
