"""One fixed normal-trained trend comparison; see docs/normal-trend-protocol.md."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.features.trends import trend_features

CHANNELS = [f"V{i}" for i in range(1, 7)]


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read_csv(payload):
    frame = pd.read_csv(io.BytesIO(payload))
    frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
    if frame.t.isna().any() or frame.t.duplicated().any() or not frame.t.is_monotonic_increasing:
        raise ValueError("Invalid, duplicated or unordered timestamps")
    return frame


def persistence(times, above):
    segments = times.diff().dt.total_seconds().gt(450).cumsum()
    return above.groupby(segments).transform(lambda s: s.rolling(3, min_periods=3).sum().eq(3))


def metrics(frame, faults, original_times, nominal_times):
    coverage = fault_window_coverage(frame.t, faults)
    original = fault_window_coverage(original_times, faults)
    events = []
    group = -1
    previous = None
    for fault, covered, raw in zip(faults, coverage, original, strict=True):
        if previous is None or fault - pd.Timedelta(hours=6) > previous:
            group += 1
        previous = fault
        window = frame[frame.t.ge(fault - pd.Timedelta(hours=6)) & frame.t.lt(fault)]
        alerts = window.loc[window.is_alert, "t"]
        events.append(
            {
                "fault": fault.isoformat(),
                "group": group,
                "original_eligible": raw["eligible"],
                "original_coverage": raw["coverage_fraction"],
                "eligible": covered["eligible"],
                "coverage": covered["coverage_fraction"],
                "warning_observed": bool(len(alerts)),
                "lead_minutes": (fault - alerts.min()).total_seconds() / 60
                if len(alerts)
                else None,
            }
        )
    eligible = [e for e in events if e["eligible"]]
    # Nominal filtering precedes episode counting; gaps break exposure and episodes.
    nominal = frame[frame.t.isin(nominal_times)].copy()
    dt = nominal.t.diff().dt.total_seconds()
    episodes = int(
        (nominal.is_alert & (~nominal.is_alert.shift(fill_value=False) | dt.gt(450))).sum()
    )
    hours = float(dt[dt.gt(0) & dt.le(450)].sum() / 3600)
    groups_warned = sorted({e["group"] for e in eligible if e["warning_observed"]})
    return {
        "score_rows": len(frame),
        "eligible_faults": len(eligible),
        "faults_warned": sum(e["warning_observed"] for e in eligible),
        "eligible_groups": len({e["group"] for e in eligible}),
        "groups_warned": groups_warned,
        "nominal_points": len(nominal),
        "nominal_alerts": int(nominal.is_alert.sum()),
        "nominal_alert_fraction": float(nominal.is_alert.mean()) if len(nominal) else None,
        "nominal_episodes": episodes,
        "observed_nominal_hours": hours,
        "episodes_per_observed_day": episodes * 24 / hours if hours else None,
        "events": events,
    }


def run(inputs: Path, diagnostics: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve earlier evidence")
    input_bytes = inputs.read_bytes()
    diagnostic_bytes = (diagnostics / "diagnostics.json").read_bytes()
    diagnostic = json.loads(diagnostic_bytes)
    if sha(input_bytes).upper() != diagnostic["input_package_sha256"].upper():
        raise ValueError("Input package does not match frozen diagnostic evidence")
    checks = json.loads((diagnostics / "checksums.json").read_text())
    for name in ("diagnostics.json", "lstm-scores.csv", "calibration-scores.csv"):
        if sha((diagnostics / name).read_bytes()).upper() != checks[name].upper():
            raise ValueError(f"Changed diagnostic artifact: {name}")
    with zipfile.ZipFile(io.BytesIO(input_bytes)) as archive:
        manifest = json.loads(archive.read("input_manifest.json"))
        if manifest["test_data_included"]:
            raise ValueError("Final-test data is forbidden in this study")
        for name, expected in manifest["files_sha256"].items():
            if sha(archive.read(name)).upper() != expected.upper():
                raise ValueError(f"Changed input: {name}")
        frames = {
            name: read_csv(archive.read(f"data/{name}.csv"))
            for name in (
                "normal_training",
                "validation_context",
                "validation_prefault",
                "validation_nominal",
                "evaluation_faults",
            )
        }
    training = frames["normal_training"]
    timeline = frames["validation_context"]
    lower, upper = map(pd.Timestamp, manifest["validation_interval"])
    if not training.t.max() < lower or not timeline.t.between(lower, upper, inclusive="left").all():
        raise ValueError("Training/development chronology changed")
    faults = frames["evaluation_faults"].t.tolist()
    if not all(lower <= f < upper for f in faults):
        raise ValueError("Faults outside development interval")
    split = int(len(training) * 0.8)
    fitting = trend_features(training.iloc[:split], CHANNELS)
    calibration = trend_features(training.iloc[split:], CHANNELS)
    development = trend_features(timeline, CHANNELS)
    if min(len(fitting), len(calibration)) < 100:
        raise ValueError("Fewer than 100 eligible normal fitting/calibration trend rows")
    columns = [c for c in fitting if c != "t"]
    means = fitting[columns].mean()
    scales = fitting[columns].std(ddof=0).clip(lower=1e-8)

    def score(features):
        return ((features[columns] - means) / scales).abs().max(axis=1)

    calibration["trend_score"] = score(calibration)
    development["trend_score"] = score(development)
    trend_threshold = float(calibration.trend_score.quantile(0.99))
    if trend_threshold <= 0:
        raise ValueError("Degenerate trend calibration threshold")
    lstm_cal = read_csv((diagnostics / "calibration-scores.csv").read_bytes())
    lstm_dev = read_csv((diagnostics / "lstm-scores.csv").read_bytes())
    lstm_threshold = float(lstm_cal.threshold.iloc[0])
    if lstm_threshold <= 0 or not lstm_cal.threshold.eq(lstm_threshold).all():
        raise ValueError("Invalid frozen LSTM threshold")
    cal = calibration.merge(lstm_cal[["t", "anomaly_score"]], on="t", validate="one_to_one")
    matched = development.merge(
        lstm_dev[["t", "anomaly_score", "is_alert"]], on="t", validate="one_to_one"
    )
    if len(cal) < 100 or matched.empty:
        raise ValueError("Insufficient matching calibration or development rows")
    cal["combined_score"] = np.maximum(
        cal.trend_score / trend_threshold, cal.anomaly_score / lstm_threshold
    )
    combined_threshold = float(cal.combined_score.quantile(0.99))
    matched["combined_score"] = np.maximum(
        matched.trend_score / trend_threshold, matched.anomaly_score / lstm_threshold
    )
    # Check causality without fitting anything on development data.
    prefix = trend_features(timeline.iloc[: len(timeline) // 2], CHANNELS)
    pd.testing.assert_frame_equal(
        prefix,
        development.loc[development.t.isin(prefix.t), ["t", *columns]].reset_index(drop=True),
    )
    bundle = {
        "columns": columns,
        "means": means.to_dict(),
        "scales": scales.to_dict(),
        "trend_threshold": trend_threshold,
        "combined_threshold": combined_threshold,
        "lstm_threshold": lstm_threshold,
    }
    replay = json.loads(json.dumps(bundle))
    np.testing.assert_allclose(
        ((development[columns] - pd.Series(replay["means"])) / pd.Series(replay["scales"]))
        .abs()
        .max(axis=1),
        development.trend_score,
    )
    policy = {
        "protocol_sha256": sha(
            (Path(__file__).resolve().parents[1] / "docs/normal-trend-protocol.md").read_bytes()
        ),
        "implementation_sha256": sha(Path(__file__).read_bytes()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_sha256": sha(input_bytes),
        "diagnostics_sha256": sha(diagnostic_bytes),
        "fit_rows": len(fitting),
        "calibration_rows": len(calibration),
        "matched_calibration_rows": len(cal),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "prefix_invariance": True,
        "statistic_replay": True,
        "test_evaluated": False,
        "model_promoted": False,
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(policy, indent=2))
    (output / "trend-model.json").write_text(json.dumps(bundle, indent=2))
    nominal_times = frames["validation_nominal"].t
    original_times = frames["validation_prefault"].t
    results = {}
    predictions = matched[["t", "trend_score", "anomaly_score", "combined_score"]].copy()
    for name, alerts in (
        ("lstm", matched.is_alert),
        ("trend", persistence(matched.t, matched.trend_score.gt(trend_threshold))),
        ("combined", persistence(matched.t, matched.combined_score.gt(combined_threshold))),
    ):
        frame = matched[["t"]].copy()
        frame["is_alert"] = alerts
        predictions[f"{name}_alert"] = alerts
        results[name] = metrics(frame, faults, original_times, nominal_times)
    results["lstm_native"] = metrics(lstm_dev, faults, original_times, nominal_times)
    baseline_groups = set(results["lstm"]["groups_warned"])
    for name in ("trend", "combined"):
        candidate = results[name]
        warned = set(candidate["groups_warned"])
        candidate["promising"] = bool(
            baseline_groups < warned
            and candidate["nominal_alert_fraction"] is not None
            and candidate["nominal_alert_fraction"] <= 0.02
        )
    predictions.to_csv(output / "predictions.csv", index=False)
    cal.to_csv(output / "calibration-scores.csv", index=False)
    summary = pd.DataFrame(
        [
            {k: v for k, v in result.items() if k != "events"} | {"model": name}
            for name, result in results.items()
        ]
    )
    events = pd.DataFrame(
        [event | {"model": name} for name, result in results.items() for event in result["events"]]
    )
    summary.to_csv(output / "summary.csv", index=False)
    events.to_csv(output / "events.csv", index=False)
    (output / "results.json").write_text(json.dumps(results, indent=2))
    hashes = {p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2))
    print(summary.to_string(index=False))
    print("Results:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("artifacts/colab/hydropredict-lstm-input-2026-10-14.zip"),
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=Path("models/lstm-diagnostics-2026-10-15-v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.diagnostics, args.output)
