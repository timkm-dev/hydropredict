"""Fixed raw/temporal comparison using validation only, with saved diagnostic plots."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import PartitionManifest, compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.features.causal import causal_feature_frame
from hydropredict.models.isolation_forest import SHPIsolationForest


def event_metrics(scores: pd.DataFrame, coverage: list[dict[str, Any]]) -> dict[str, Any]:
    alert_times = pd.to_datetime(scores.loc[scores.is_alert, "t"], utc=True)
    events = []
    for row in coverage:
        event = dict(row)
        fault = pd.Timestamp(event["fault_time"])
        preceding = alert_times[
            (alert_times < fault) & (alert_times >= fault - pd.Timedelta(hours=6))
        ]
        event["detected"] = bool(len(preceding))
        event["lead_minutes"] = (
            (fault - preceding.min()).total_seconds() / 60 if len(preceding) else None
        )
        window = scores[(scores.t < fault) & (scores.t >= fault - pd.Timedelta(hours=6))]
        event["max_score"] = float(window.anomaly_score.max()) if len(window) else None
        events.append(event)
    eligible = [e for e in events if e["eligible"]]
    detected = [e for e in eligible if e["detected"]]
    return {
        "eligible_events": len(eligible),
        "detected_events": len(detected),
        "recall": len(detected) / len(eligible) if eligible else None,
        "median_lead_minutes": float(np.median([e["lead_minutes"] for e in detected]))
        if detected
        else None,
        "events": events,
    }


def nominal_metrics(scores: pd.DataFrame) -> dict[str, Any]:
    gaps = scores.t.diff().dt.total_seconds()
    new_episode = ~scores.is_alert.shift(fill_value=False) | gaps.gt(450)
    seconds = float(gaps[(gaps > 0) & (gaps <= 450)].sum())
    episodes = int((scores.is_alert & new_episode).sum())
    return {
        "point_alert_fraction": float(scores.is_alert.mean()),
        "point_alerts": int(scores.is_alert.sum()),
        "observations": len(scores),
        "alert_episodes": episodes,
        "observed_hours": seconds / 3600,
        "episodes_per_observed_day": episodes / (seconds / 86400) if seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve prior experiments and choose a new directory.")
    root = Path(__file__).resolve().parents[1]
    manifest = PartitionManifest.load(args.preparation_run / "partitions/partitions_manifest.json")
    if not manifest.is_strictly_chronological:
        raise ValueError("Chronological partitions required.")
    frames = {}
    for name in ("train", "validation"):
        part = manifest.partitions[name]
        if compute_file_sha256(part.relative_path) != part.sha256:
            raise ValueError(f"Partition checksum changed: {name}")
        frames[name] = load_telemetry_dataframe(part.relative_path)
    split = int(len(frames["train"]) * 0.8)
    fitting = frames["train"].iloc[:split].copy()
    calibration = frames["train"].iloc[split:].copy()
    validation = frames["validation"]
    lower, upper = pd.Timestamp("2019-02-01", tz="UTC"), pd.Timestamp("2019-05-01", tz="UTC")
    if (
        calibration.t.max() >= lower
        or not validation.t.between(lower, upper, inclusive="left").all()
    ):
        raise ValueError("This fixed development protocol requires February-April validation.")
    healthy_path = root / "data/raw/shp_train.csv"
    source_hashes = manifest.source_checksums
    if compute_file_sha256(healthy_path) not in source_hashes.values():
        raise ValueError("Normal-reference source changed.")
    nominal_all = load_telemetry_dataframe(healthy_path)
    nominal = nominal_all[(nominal_all.t >= lower) & (nominal_all.t < upper)].copy()
    faults = [f for f in load_fault_log(root / "data/raw/shp_faults.csv") if lower <= f < upper]
    coverage = fault_window_coverage(validation.t, faults)
    common_validation = causal_feature_frame(validation, "temporal").t
    common_nominal = causal_feature_frame(nominal, "temporal").t
    original = SHPIsolationForest.load(args.baseline)
    original_scores = original.score(validation)
    args.output.mkdir(parents=True)
    policy = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "modes": ["raw", "temporal"],
        "quantiles": [0.99, 0.975, 0.95],
        "seed": 42,
        "trees": 200,
        "max_samples": 256,
        "selection": "Compare eligible recall within 2% nominal point-alert budget; no promotion",
        "test_partition_read": False,
        "temporal": "36 inputs; 30/120min mean/std, rates; 450s reset, 2h warmup; no fill",
        "coverage": "Fixed original policy; scored-window coverage and warmup loss also reported",
        "nominal_caveat": "Selected normal reference; not general operational false alarms",
    }
    (args.output / "protocol.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    original_threshold = float(original.threshold) if original.threshold is not None else 0
    diagnostics = []
    means, stds = original.scaler.means, original.scaler.stds
    for number, event in enumerate(event_metrics(original_scores, coverage)["events"], start=1):
        fault = pd.Timestamp(event["fault_time"])
        window = validation[
            (validation.t >= fault - pd.Timedelta(hours=12)) & (validation.t < fault)
        ]
        window_scores = original_scores[original_scores.t.isin(window.t)]
        diagnosis = dict(event)
        diagnosis["threshold"] = original_threshold
        maximum = diagnosis["max_score"]
        diagnosis["margin_below_threshold"] = (
            original_threshold - maximum if maximum is not None else None
        )
        for label, start_hour, end_hour in (("early", -6, -3), ("late", -1, 0)):
            relative = (window_scores.t - fault).dt.total_seconds() / 3600
            selected = window_scores.loc[(relative >= start_hour) & (relative < end_hour)]
            diagnosis[f"{label}_median_score"] = (
                float(selected.anomaly_score.median()) if len(selected) else None
            )
        diagnosis["interpretation"] = (
            "insufficient coverage"
            if not event["eligible"]
            else "threshold not crossed"
            if not event["detected"]
            else "detected"
        )
        diagnostics.append(diagnosis)
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for col in means:
            axes[0].plot(
                (window.t - fault).dt.total_seconds() / 3600,
                (window[col] - means[col]) / stds[col],
                label=col,
                linestyle="none",
                marker=".",
                markersize=2,
            )
        axes[0].set_ylabel("Standardised channel value")
        axes[0].legend(ncol=6, fontsize=8)
        axes[1].plot(
            (window_scores.t - fault).dt.total_seconds() / 3600,
            window_scores.anomaly_score,
            label="Original anomaly score",
            linestyle="none",
            marker=".",
            markersize=3,
        )
        axes[1].axhline(original_threshold, color="red", linestyle="--", label="Frozen threshold")
        for axis in axes:
            axis.axvline(0, color="black", linestyle=":")
            axis.axvspan(-6, 0, color="orange", alpha=0.08)
            axis.grid(alpha=0.2)
        axes[1].set_xlabel("Hours before recorded fault")
        axes[1].set_ylabel("Anomaly score")
        axes[1].legend(fontsize=8)
        figure.suptitle(f"Validation event {number}: {fault} | {diagnosis['interpretation']}")
        figure.tight_layout()
        figure.savefig(args.output / f"validation_event_{number:02d}.png", dpi=140)
        plt.close(figure)
    (args.output / "missed_fault_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    results = []
    for mode in policy["modes"]:
        model = SHPIsolationForest(feature_mode=mode)
        model.fit(fitting, calibration, version=f"validation-{mode}")
        cal_scores = model.score(calibration).anomaly_score.to_numpy()
        for quantile in policy["quantiles"]:
            model.threshold = float(np.quantile(cal_scores, quantile))
            model.version = f"{args.output.name}-{mode}-q{quantile}"
            model.save(args.output / f"{mode}-q{quantile}", policy)
            reloaded = SHPIsolationForest.load(args.output / f"{mode}-q{quantile}")
            scored = reloaded.score(validation)
            normal_scores = reloaded.score(nominal)
            np.testing.assert_allclose(model.score(validation).anomaly_score, scored.anomaly_score)
            if mode == "raw" and quantile == 0.99:
                np.testing.assert_allclose(original_scores.anomaly_score, scored.anomaly_score)
            paired = scored[scored.t.isin(common_validation)]
            paired_normal = normal_scores[normal_scores.t.isin(common_nominal)]
            result = {
                "mode": mode,
                "quantile": quantile,
                "threshold": model.threshold,
                "score_rows": len(scored),
                "input_rows": len(validation),
                "warmup_excluded_rows": len(validation) - len(scored),
                "all_available": event_metrics(scored, coverage),
                "paired": event_metrics(paired, coverage),
                "scored_coverage": fault_window_coverage(scored.t, faults),
                "nominal": nominal_metrics(normal_scores),
                "paired_nominal": nominal_metrics(paired_normal),
            }
            scored_eligible = {e["fault_time"] for e in result["scored_coverage"] if e["eligible"]}
            result["effective_coverage_detected_events"] = sum(
                e["detected"] and e["fault_time"] in scored_eligible
                for e in result["paired"]["events"]
            )
            results.append(result)
            scored.to_csv(args.output / f"{mode}-q{quantile}-validation.csv", index=False)
            normal_scores.to_csv(args.output / f"{mode}-q{quantile}-nominal.csv", index=False)
            print(
                mode,
                quantile,
                "paired detections",
                result["paired"]["detected_events"],
                "nominal fraction",
                result["paired_nominal"]["point_alert_fraction"],
            )
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for mode in policy["modes"]:
        selected = [r for r in results if r["mode"] == mode]
        x = [100 * r["paired_nominal"]["point_alert_fraction"] for r in selected]
        y = [r["paired"]["detected_events"] for r in selected]
        axis.plot(x, y, marker="o", label=mode)
        for a, b, result in zip(x, y, selected, strict=True):
            axis.annotate(
                f"q={result['quantile']}",
                (a, b),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axvline(2, color="gray", linestyle="--", label="2% development alert budget")
    axis.set_xlabel("Point alerts on matched selected normal-reference readings (%)")
    axis.set_ylabel("Warnings before originally eligible validation events (of 6)")
    axis.set_title("Validation sensitivity and alert burden")
    axis.set_ylim(-0.2, 6.5)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.text(
        0.5,
        0.02,
        "Temporal warm-up: 400/605 readings; only 2 events retain full scored coverage. "
        "No test evaluation.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(args.output / "validation_tradeoff.png", dpi=140)
    plt.close(figure)
    hashes = {p.name: compute_file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
