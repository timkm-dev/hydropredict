"""Execute the fixed missed-fault study on checksum-verified development inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import zipfile
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from compare_validation_models import event_metrics, nominal_metrics

from hydropredict.analysis.novelty import CANDIDATES, NormalNovelty
from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.data.schema import MEASUREMENT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def read_frame(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
    return frame


def enrich_events(
    scores: pd.DataFrame, coverage: list[dict[str, Any]], timeline: pd.DataFrame
) -> dict[str, Any]:
    result = event_metrics(scores, coverage)
    gaps = timeline.t.diff().dt.total_seconds()
    starts = timeline.t.where(gaps.isna() | gaps.gt(450)).ffill()
    age = pd.Series((timeline.t - starts).dt.total_seconds().to_numpy(), index=timeline.t)
    for event in result["events"]:
        end = pd.Timestamp(event["fault_time"])
        window = scores[(scores.t >= end - pd.Timedelta(hours=6)) & (scores.t < end)]
        positives = window[window.is_alert]
        event["positive_readings"] = len(positives)
        event["last_lead_minutes"] = (
            (end - positives.t.max()).total_seconds() / 60 if len(positives) else None
        )
        event["first_warning_context_minutes"] = (
            float(age.loc[positives.t.min()]) / 60 if len(positives) else None
        )
        if len(window):
            event["max_threshold_ratio"] = float((window.anomaly_score / window.threshold).max())
        else:
            event["max_threshold_ratio"] = None
    return result


def channel_diagnostics(
    fitting: pd.DataFrame,
    calibration: pd.DataFrame,
    timeline: pd.DataFrame,
    faults: list[pd.Timestamp],
    output: Path,
) -> None:
    columns = list(MEASUREMENT_COLUMNS)
    fitting[columns].describe(percentiles=[0.01, 0.5, 0.99]).to_csv(
        output / "fit-channel-summary.csv"
    )
    fitting[columns].corr().to_csv(output / "fit-channel-correlations.csv")
    means, scales = fitting[columns].mean(), fitting[columns].std(ddof=0).clip(lower=1e-8)
    lower, upper = calibration[columns].quantile(0.001), calibration[columns].quantile(0.999)
    records = []
    for index, fault in enumerate(faults, 1):
        window = timeline[(timeline.t < fault) & (timeline.t >= fault - pd.Timedelta(hours=6))]
        figure, axes = plt.subplots(6, 1, figsize=(11, 10), sharex=True)
        for channel, axis in zip(columns, axes, strict=True):
            values = window[channel]
            records.append(
                {
                    "fault_time": fault.isoformat(),
                    "channel": channel,
                    "observations": len(values),
                    "median_fit_z": float((values.median() - means[channel]) / scales[channel])
                    if len(values)
                    else None,
                    "calibration_extreme_readings": int(
                        ((values < lower[channel]) | (values > upper[channel])).sum()
                    ),
                    "min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                }
            )
            hours = (window.t - fault).dt.total_seconds() / 3600
            series = values.copy()
            # Do not visually bridge outages.
            series.loc[window.t.diff().dt.total_seconds().gt(450)] = np.nan
            axis.plot(hours, series, color="#227f73", lw=1)
            axis.axhspan(lower[channel], upper[channel], color="#adc5ba", alpha=0.25)
            axis.set_ylabel(channel)
            axis.grid(alpha=0.15)
        axes[-1].set_xlabel("Hours before recorded fault (UTC); gaps are not filled")
        figure.suptitle(
            f"Development fault {index}: {fault.isoformat()}\n"
            "Shading: normal calibration 0.1–99.9 percentiles; channel meanings unverified"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(output / f"raw-event-{index:02d}.png", dpi=130)
        plt.close(figure)
    pd.DataFrame(records).to_csv(output / "event-channel-diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "artifacts/colab/hydropredict-lstm-input-2026-10-14.zip",
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=ROOT / "models/lstm-diagnostics-2026-10-15-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new directory; preserve prior evidence.")
    diagnostics = json.loads((args.diagnostics / "diagnostics.json").read_text())
    diagnostic_hashes = json.loads((args.diagnostics / "checksums.json").read_text())
    for name in (
        "diagnostics.json",
        "lstm-scores.csv",
        "if_original_raw-scores.csv",
        "if_30min-scores.csv",
    ):
        if compute_file_sha256(args.diagnostics / name) != diagnostic_hashes[name]:
            raise ValueError(f"Diagnostic input changed: {name}")
    if compute_file_sha256(args.inputs) != diagnostics["input_package_sha256"]:
        raise ValueError("Original input package hash changed.")
    with zipfile.ZipFile(args.inputs) as archive:
        manifest = json.loads(archive.read("input_manifest.json"))
        frames = {}
        for name in (
            "normal_training",
            "validation_context",
            "validation_nominal",
            "validation_prefault",
            "evaluation_faults",
        ):
            key = f"data/{name}.csv"
            payload = archive.read(key)
            if hashlib.sha256(payload).hexdigest().upper() != manifest["files_sha256"][key]:
                raise ValueError(f"Packaged input changed: {key}")
            frames[name] = read_frame(payload)
    training, timeline = frames["normal_training"], frames["validation_context"]
    split = int(len(training) * 0.8)
    fitting, calibration = training.iloc[:split], training.iloc[split:]
    start, end = map(pd.Timestamp, manifest["validation_interval"])
    if calibration.t.max() >= start or not timeline.t.between(start, end, inclusive="left").all():
        raise ValueError("Partition boundary changed.")
    if manifest["test_data_included"] or len(training) != 30807:
        raise ValueError("Declared training scope changed.")
    faults = frames["evaluation_faults"].t.tolist()
    if not all(start <= fault < end for fault in faults):
        raise ValueError("Fault labels extend beyond development interval.")
    original_coverage = fault_window_coverage(frames["validation_prefault"].t, faults)
    nominal_times = set(frames["validation_nominal"].t)
    args.output.mkdir(parents=True)
    protocol = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol_sha256": compute_file_sha256(ROOT / "docs/signal-investigation-protocol.md"),
        "input_package_sha256": compute_file_sha256(args.inputs),
        "diagnostics_sha256": diagnostic_hashes["diagnostics.json"],
        "candidates": list(CANDIDATES),
        "fit_rows": len(fitting),
        "calibration_rows": len(calibration),
        "interval": [start.isoformat(), end.isoformat()],
        "test_evaluated": False,
        "promoted": False,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    channel_diagnostics(fitting, calibration, timeline, faults, args.output)
    scored = {
        name: read_frame((args.diagnostics / f"{name}-scores.csv").read_bytes())
        for name in ("if_original_raw", "if_30min", "lstm")
    }
    model_details = {}
    for method in CANDIDATES:
        begun = perf_counter()
        model = NormalNovelty(method).fit(fitting, calibration)
        scores = model.score(timeline)
        prefix = model.score(timeline.iloc[:200])
        np.testing.assert_allclose(
            prefix.anomaly_score, scores[scores.t.isin(prefix.t)].anomaly_score, rtol=1e-10
        )
        bundle = args.output / f"{method}.joblib"
        joblib.dump(model, bundle)
        restored = joblib.load(bundle)
        pd.testing.assert_frame_equal(scores, restored.score(timeline))
        scored[method] = scores
        model_details[method] = {
            "fit_rows": model.fit_rows,
            "calibration_rows": model.calibration_rows,
            "threshold": model.threshold,
            "seconds": perf_counter() - begun,
        }
        if method == "pca_residual":
            model_details[method]["explained_variance_ratio"] = (
                model.estimator.explained_variance_ratio_.tolist()
            )
        if method == "regime_covariance":
            occupancy = []
            for name, frame in (
                ("fitting", fitting),
                ("calibration", calibration),
                ("development", timeline),
            ):
                labels = model.estimator.predict(model.matrix(model.features(frame)))
                for label in range(8):
                    occupancy.append(
                        {
                            "block": name,
                            "regime": label,
                            "readings": int((labels == label).sum()),
                            "fraction": float((labels == label).mean()),
                        }
                    )
            pd.DataFrame(occupancy).to_csv(args.output / "regime-occupancy.csv", index=False)
        print(f"Scored {method}: {len(scores)} observations", flush=True)
    common = set.intersection(*(set(frame.t) for frame in scored.values()))
    common_coverage = fault_window_coverage(pd.Series(sorted(common)), faults)
    results, overview, event_rows, monthly_rows = {}, [], [], []
    for name, scores in scored.items():
        matched = scores[scores.t.isin(common)]
        nominal = matched[matched.t.isin(nominal_times)]
        events = enrich_events(matched, common_coverage, timeline)
        burden = nominal_metrics(nominal)
        original = enrich_events(matched, original_coverage, timeline)
        result = {
            "matched_adequately_scored": events,
            "matched_original_eligible": original,
            "native_adequately_scored": enrich_events(
                scores, fault_window_coverage(scores.t, faults), timeline
            ),
            "matched_nominal": burden,
            "native_nominal": nominal_metrics(scores[scores.t.isin(nominal_times)]),
            "scored_rows": len(scores),
            "unscored_rows": len(timeline) - len(scores),
            "promising_development_candidate": name in CANDIDATES
            and events["detected_events"] >= 2
            and burden["point_alert_fraction"] <= 0.02,
        }
        results[name] = result
        scores.to_csv(args.output / f"{name}-scores.csv", index=False)
        overview.append(
            {
                "model": name,
                "detected": events["detected_events"],
                "eligible": events["eligible_events"],
                "original_detected": original["detected_events"],
                "original_eligible": original["eligible_events"],
                "nominal_alerts": burden["point_alerts"],
                "nominal_rows": burden["observations"],
                "nominal_fraction": burden["point_alert_fraction"],
                "nominal_episodes": burden["alert_episodes"],
                "scored_rows": len(scores),
                "promising": result["promising_development_candidate"],
            }
        )
        event_rows.extend({"model": name, **event} for event in events["events"])
        for month, group in nominal.groupby(nominal.t.dt.strftime("%Y-%m")):
            monthly_rows.append({"model": name, "month": month, **nominal_metrics(group)})
    pd.DataFrame(overview).to_csv(args.output / "comparison.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.output / "events.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(args.output / "monthly-nominal.csv", index=False)
    payload = {
        "models": results,
        "fitting": model_details,
        "common_rows": len(common),
        "common_nominal_rows": len(common & nominal_times),
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    # Complete event matrix retains ineligible faults instead of hiding missing coverage.
    matrix = (
        pd.DataFrame(event_rows)
        .pivot(index="model", columns="fault_time", values="max_threshold_ratio")
        .reindex(scored)
    )
    figure, axis = plt.subplots(figsize=(14, 5))
    image = axis.imshow(
        np.minimum(matrix.to_numpy(dtype=float), 3), vmin=0, vmax=3, cmap="YlOrRd", aspect="auto"
    )
    axis.set_yticks(range(len(matrix)), matrix.index)
    axis.set_xticks(
        range(len(matrix.columns)),
        [f"{pd.Timestamp(t):%b %d %H:%M}" for t in matrix.columns],
        rotation=30,
        ha="right",
    )
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title(
        "Development peak score / threshold before each fault\n"
        ">1 crosses threshold; eligibility and shared-warning caveats apply"
    )
    figure.colorbar(image, ax=axis, label="Peak / threshold (colour clipped at 3)")
    figure.tight_layout()
    figure.savefig(args.output / "event-score-comparison.png", dpi=150)
    plt.close(figure)
    hashes = {
        str(p.relative_to(args.output)): compute_file_sha256(p)
        for p in args.output.rglob("*")
        if p.is_file()
    }
    (args.output / "checksums.json").write_text(json.dumps(hashes, indent=2))
    print(pd.DataFrame(overview).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
