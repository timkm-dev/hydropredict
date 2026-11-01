"""Reproduce illustrative CARE timelines from saved scores without retraining."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    out = Path(".local/care-mentor-report")
    out.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv("docs/results/care-residual-v2/events.csv")
    inventory = json.loads(Path("docs/results/care-eda-2026-10-28/inventory.json").read_text())
    selected = [(19, "Detected anomaly"), (7, "Missed anomaly"), (2, "Normal control with alarms")]
    audit = []
    for event_id, title in selected:
        row = records[(records.farm == "Wind Farm B") & (records.event_id == event_id)].iloc[0]
        info = next(
            i for i in inventory if i["farm"] == "Wind Farm B" and i["event_id"] == event_id
        )
        scores = pd.read_csv(f".local/care-residual-v2-models/Wind_Farm_B_{event_id}_scores.csv")
        ts = pd.to_datetime(scores.time)
        days = (ts - ts.iloc[0]).dt.total_seconds() / 86400
        ratio = scores.score / row.threshold
        fig, ax = plt.subplots(figsize=(7, 1.85))
        ax.plot(days, ratio, color="#205d78", linewidth=0.65, label="Score / threshold")
        ax.axhline(1, color="#b84737", linewidth=0.85, linestyle="--", label="Alarm threshold")
        if row.label == "anomaly":
            start = (pd.Timestamp(info["event_start"]) - ts.iloc[0]).total_seconds() / 86400
            end = (pd.Timestamp(info["event_end"]) - ts.iloc[0]).total_seconds() / 86400
            ax.axvspan(start, end, color="#e8b85d", alpha=0.23, label="Labelled anomaly interval")
            ax.axvline(
                end - 1,
                color="#777777",
                linewidth=0.7,
                linestyle=":",
                label="24 hours before interval end",
            )
        alerts = scores.alert.astype(bool)
        ax.scatter(
            days[alerts], ratio[alerts], s=5, color="#b84737", label="Persistent alert", zorder=3
        )
        ax.set_title(f"{title}  |  Farm B event {event_id}", loc="left", fontsize=10)
        ax.set_xlabel("Days since prediction period began", fontsize=8)
        ax.set_ylabel("Relative score", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(days.iloc[0], days.iloc[-1])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout(pad=0.7)
        fig.savefig(out / f"case-{event_id}.png", dpi=200)
        plt.close(fig)
        audit.append(
            dict(
                event_id=event_id,
                label=row.label,
                episodes=int(row.episodes),
                alert_rows=int(row.alert_rows),
                prediction_rows=int(row.prediction_rows),
                detected_24h=bool(row.detected_24h),
                lead_hours=None
                if pd.isna(row.end_relative_lead_hours)
                else float(row.end_relative_lead_hours),
                description=row.description,
            )
        )
    (out / "cases.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
