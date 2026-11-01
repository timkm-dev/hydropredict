"""Descriptive diagnostics of earlier CARE outputs; does not tune any detector."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    out = Path("docs/results/care-supervised-v3")
    rows = []
    for version, folder in [("v1", "care-baseline-v1"), ("v2", "care-residual-v2")]:
        events = pd.read_csv(Path("docs/results") / folder / "events.csv")
        inventory = json.loads(Path("docs/results/care-eda-2026-10-28/inventory.json").read_text())
        for r in events.itertuples():
            name = f"{r.farm.replace(' ', '_')}_{r.event_id}_scores.csv"
            score = pd.read_csv(Path(".local") / (folder + "-models") / name)
            times = pd.to_datetime(score.time)
            info = next(i for i in inventory if i["farm"] == r.farm and i["event_id"] == r.event_id)
            mask = times.between(pd.Timestamp(info["event_start"]), pd.Timestamp(info["event_end"]))
            if r.label == "normal":
                mask = np.ones(len(score), dtype=bool)
            selected = score.loc[mask]
            scaled = selected.score / r.threshold
            rows.append(
                dict(
                    version=version,
                    farm=r.farm,
                    event_id=r.event_id,
                    asset=r.asset,
                    label=r.label,
                    relative_score_p90=float(scaled.quantile(0.9)),
                    point_exceedance_fraction=float(selected.score.gt(r.threshold).mean()),
                    persistent_alert_fraction=float(selected.alert.mean()),
                )
            )
    data = pd.DataFrame(rows)
    data.to_csv(out / "earlier-score-diagnostics.csv", index=False)
    summary = []
    for (version, farm), g in data.groupby(["version", "farm"]):
        summary.append(
            dict(
                version=version,
                farm=farm,
                descriptive_file_auc=float(
                    roc_auc_score(g.label.eq("anomaly"), g.relative_score_p90)
                ),
                normal_median_p90=float(g[g.label.eq("normal")].relative_score_p90.median()),
                anomaly_median_p90=float(g[g.label.eq("anomaly")].relative_score_p90.median()),
            )
        )
    (out / "earlier-score-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
