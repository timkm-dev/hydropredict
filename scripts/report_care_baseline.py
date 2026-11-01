"""Render aggregate CARE baseline evidence without selecting models or thresholds."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path("docs/results/care-baseline-v1")
    summary = json.loads((root / "summary.json").read_text())
    events = pd.read_csv(root / "events.csv")
    assert len(events) == 37 and events.event_id.nunique() == 37
    for row in events.itertuples():
        assert pd.Timestamp(row.fit_end) < pd.Timestamp(row.calibration_start)
        assert pd.Timestamp(row.calibration_end) < pd.Timestamp(row.prediction_start)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["Farm A", "Farm B"]
    values = [
        [100 * s["detected_24h"] / s["anomaly_files"] for s in summary],
        [s["normal_episodes_per_30_days"] for s in summary],
        [100 * s["normal_alert_fraction"] for s in summary],
    ]
    titles = [
        "Detection >=24h before interval end (%)",
        "Normal alarms / 30 observation-days",
        "Normal readings alerted (%)",
    ]
    for ax, vals, title, gate in zip(axes, values, titles, [50, 1, 1], strict=True):
        ax.bar(labels, vals, color=["#147d92", "#d77b38"])
        ax.axhline(gate, color="#9a2933", linestyle="--", label="Frozen screening gate")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(vals) * 1.25, gate * 1.5))
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.suptitle("CARE: fixed Isolation Forest pilot (no outcome tuning)")
    fig.tight_layout()
    fig.savefig(root / "baseline-results.png", dpi=160)
    print(
        events.loc[
            events.label.eq("anomaly"),
            [
                "farm",
                "event_id",
                "asset",
                "description",
                "detected_24h",
                "end_relative_lead_hours",
                "preexisting_alert",
            ],
        ].to_string(index=False)
    )
    print("normal controls by turbine:")
    print(
        events.loc[events.label.eq("normal")]
        .groupby(["farm", "asset"])
        .agg(
            files=("event_id", "size"), episodes=("episodes", "sum"), days=("observed_days", "sum")
        )
        .to_string()
    )
    print("Total seconds:", json.loads((root / "config.json").read_text())["elapsed_seconds"])


if __name__ == "__main__":
    main()
