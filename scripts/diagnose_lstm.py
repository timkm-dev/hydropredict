"""Diagnose a frozen LSTM export against frozen local development comparators."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from compare_validation_models import event_metrics, nominal_metrics

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.models.isolation_forest import SHPIsolationForest
from hydropredict.models.lstm import SHPLSTMAutoencoder


def read_frame(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; previous evidence is preserved.")
    original_hash = compute_file_sha256(args.export)
    with zipfile.ZipFile(args.inputs) as package:
        manifest = json.loads(package.read("input_manifest.json"))
        payloads = {}
        for name in (
            "normal_training.csv",
            "validation_context.csv",
            "validation_nominal.csv",
            "validation_prefault.csv",
            "evaluation_faults.csv",
        ):
            key = f"data/{name}"
            payload = package.read(key)
            if hashlib.sha256(payload).hexdigest().upper() != manifest["files_sha256"][key]:
                raise ValueError(f"Input checksum mismatch: {key}")
            payloads[name] = payload
    with zipfile.ZipFile(args.export) as archive:
        metadata_name = next(n for n in archive.namelist() if n.endswith("/bundle/metadata.json"))
        prefix = metadata_name.removesuffix("bundle/metadata.json")
        metadata = json.loads(archive.read(metadata_name))
        provenance = metadata["provenance"]["provenance"]
        if provenance["source_commit"] != manifest["source_commit"] or (
            provenance["training_input_hash"]
            != manifest["files_sha256"]["data/normal_training.csv"]
        ):
            raise ValueError("Export does not match packaged source/training inputs.")
        exported = read_frame(archive.read(prefix + "validation/scores.csv"))
        args.output.mkdir(parents=True)
        bundle = args.output / "frozen_bundle_copy"
        bundle.mkdir()
        # Only copy known bundle members; archive paths never select destination paths.
        for name in ("weights.pt", "scaler.json", "metadata.json", "checksums.json"):
            (bundle / name).write_bytes(archive.read(prefix + "bundle/" + name))
    torch.set_num_threads(1)
    lstm = SHPLSTMAutoencoder.load(bundle)
    timeline = read_frame(payloads["validation_context.csv"])
    nominal_times = read_frame(payloads["validation_nominal.csv"]).t
    prefault = read_frame(payloads["validation_prefault.csv"])
    faults = list(read_frame(payloads["evaluation_faults.csv"]).t)
    lower, upper = map(pd.Timestamp, manifest["validation_interval"])
    if not timeline.t.between(lower, upper, inclusive="left").all():
        raise ValueError("Development interval violation.")
    training = read_frame(payloads["normal_training.csv"])
    calibration = training.iloc[int(len(training) * 0.8) :]
    if calibration.t.max() >= lower:
        raise ValueError("Calibration overlaps development validation.")
    scores = lstm.score(timeline)
    if not scores.t.equals(exported.t):
        raise ValueError("Replayed timestamps differ from export.")
    error_columns = [f"V{i}_reconstruction_error" for i in range(1, 7)]
    np.testing.assert_allclose(
        scores[["anomaly_score", *error_columns]],
        exported[["anomaly_score", *error_columns]],
        rtol=1e-4,
        atol=1e-6,
    )
    if not np.array_equal(scores.is_alert, exported.is_alert):
        raise ValueError("CPU replay changed exported alert decisions.")
    models = {"lstm": scores}
    paths = {
        "if_original_raw": Path("models/isolation-forest-2026-10-13-v1/bundle"),
        "if_30min": Path("models/short-context-2026-10-14-v1/model-30min"),
    }
    # Original baseline directories may themselves be bundles.
    if not paths["if_original_raw"].exists():
        paths["if_original_raw"] = paths["if_original_raw"].parent
    for name, path in paths.items():
        models[name] = SHPIsolationForest.load(path).score(timeline)
    common = set.intersection(*(set(frame.t) for frame in models.values()))
    common_nominal = common.intersection(set(nominal_times))
    coverage = fault_window_coverage(prefault.t, faults)
    matched_coverage = fault_window_coverage(pd.Series(sorted(common)), faults)
    report = {
        "status": "Development diagnostics only; no final test evaluation or model promotion",
        "export_sha256": original_hash,
        "input_package_sha256": compute_file_sha256(args.inputs),
        "source_commit": manifest["source_commit"],
        "replay_verified": True,
        "test_evaluated": False,
        "nominal_caveat": "Selected normal reference; not real-world false-alarm rates",
        "common_score_rows": len(common),
        "common_nominal_rows": len(common_nominal),
        "models": {},
    }
    for name, frame in models.items():
        matched = frame[frame.t.isin(common)]
        report["models"][name] = {
            "native_original_eligible": event_metrics(frame, coverage),
            "native_adequately_scored": event_metrics(
                frame, fault_window_coverage(frame.t, faults)
            ),
            "matched_original_eligible": event_metrics(matched, coverage),
            "matched_adequately_scored": event_metrics(matched, matched_coverage),
            "matched_nominal": nominal_metrics(frame[frame.t.isin(common_nominal)]),
            "native_nominal": nominal_metrics(frame[frame.t.isin(nominal_times)]),
        }
        frame.to_csv(args.output / f"{name}-scores.csv", index=False)
    nominal = scores[scores.t.isin(nominal_times)]
    nominal[["anomaly_score", *error_columns]].quantile([0.5, 0.9, 0.95, 0.99]).to_csv(
        args.output / "normal-error-quantiles.csv"
    )
    diagnostics = []
    for index, event in enumerate(report["models"]["lstm"]["native_original_eligible"]["events"]):
        fault = pd.Timestamp(event["fault_time"])
        window = scores[(scores.t >= fault - pd.Timedelta(hours=6)) & (scores.t < fault)]
        row = dict(event)
        row["max_score_over_threshold"] = (
            float(window.anomaly_score.max() / lstm.threshold) if len(window) else None
        )
        row["channels"] = {}
        for column in error_columns:
            maximum = float(window[column].max()) if len(window) else None
            row["channels"][column] = {
                "max_error": maximum,
                "normal_percentile_of_max": float((nominal[column] <= maximum).mean())
                if maximum is not None
                else None,
                "fraction_above_normal_q99": float(
                    (window[column] > nominal[column].quantile(0.99)).mean()
                )
                if len(window)
                else None,
            }
        diagnostics.append(row)
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        raw = timeline[(timeline.t >= fault - pd.Timedelta(hours=6)) & (timeline.t < fault)]
        scaled = lstm.scaler.transform(raw)
        for channel in range(1, 7):
            col = f"V{channel}"
            axes[0].plot(scaled.t, scaled[col], ".", ms=3, label=col)
            axes[2].plot(window.t, window[f"{col}_reconstruction_error"], ".", ms=3, label=col)
        for name, frame in models.items():
            view = frame[(frame.t >= fault - pd.Timedelta(hours=6)) & (frame.t < fault)]
            axes[1].plot(view.t, view.anomaly_score / view.threshold, ".", ms=3, label=name)
        axes[1].axhline(1, color="red", ls="--")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(ncol=3, fontsize=8)
        axes[0].set_ylabel("Standardised reading")
        axes[1].set_ylabel("Score / model threshold")
        axes[2].set_ylabel("Channel reconstruction MSE")
        axes[2].set_xlabel("UTC; points only, gaps are not interpolated")
        fig.suptitle(f"Fault {fault} | original eligible={event['eligible']}")
        fig.tight_layout()
        fig.savefig(args.output / f"event-{index + 1:02d}.png", dpi=140)
        plt.close(fig)
    pd.DataFrame([{k: v for k, v in row.items() if k != "channels"} for row in diagnostics]).to_csv(
        args.output / "event-summary.csv", index=False
    )
    report["event_diagnostics"] = diagnostics
    first_fault = faults[0]
    isolated = scores[(scores.t >= first_fault - pd.Timedelta(hours=6)) & (scores.t < first_fault)]
    isolated.to_csv(args.output / "february-2-warning-context.csv", index=False)
    report["first_fault_warning"] = {
        "alert_rows": int(isolated.is_alert.sum()),
        "first_exported_score_is_alert": bool(scores.iloc[0].is_alert),
        "first_score": str(scores.iloc[0].t),
        "first_window_start": str(scores.iloc[0].window_start),
        "dominant_channel": scores.iloc[0][error_columns].astype(float).idxmax(),
    }
    calibration_scores = lstm.score(calibration)
    calibration_scores.to_csv(args.output / "calibration-scores.csv", index=False)
    report["calibration_threshold_reproduced"] = float(
        np.quantile(calibration_scores.anomaly_score, lstm.config.calibration_quantile)
    )
    np.testing.assert_allclose(
        report["calibration_threshold_reproduced"], lstm.threshold, rtol=1e-4
    )
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if compute_file_sha256(args.export) != original_hash:
        raise ValueError("Original export changed.")
    checksums = {
        str(path.relative_to(args.output)): compute_file_sha256(path)
        for path in args.output.rglob("*")
        if path.is_file()
    }
    (args.output / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("models", "event_diagnostics")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
