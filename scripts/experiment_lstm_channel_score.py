"""One fixed calibration-only channel scoring experiment; no weight updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_validation_models import event_metrics, nominal_metrics

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.partitioning import compute_file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior experiments; choose a new output directory.")
    hashes = json.loads((args.diagnostics / "checksums.json").read_text())
    for name, expected in hashes.items():
        if compute_file_sha256(args.diagnostics / name) != expected:
            raise ValueError(f"Diagnostic input changed: {name}")
    protocol = {
        "hypothesis": "Averaging six channel errors may dilute isolated channel deviations",
        "score": "maximum over channels of error / channel calibration q0.99",
        "threshold": "q0.99 of joint maximum score on original normal calibration sequences",
        "weights_updated": False,
        "channel_selection": "All six channels equally; no fault-specific weighting",
        "search": "One candidate only; no threshold grid or post-result tuning",
        "decision": "Additional adequately scored detection and <=2% matched normal point alerts",
        "test_evaluated": False,
        "caveat": "Development experiment; normal-reference alerts are not real-world false alarms",
        "diagnostics_sha256": compute_file_sha256(args.diagnostics / "diagnostics.json"),
    }
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    calibration = pd.read_csv(args.diagnostics / "calibration-scores.csv")
    columns = [f"V{i}_reconstruction_error" for i in range(1, 7)]
    scales = calibration[columns].quantile(0.99).clip(lower=np.finfo(float).eps)
    threshold = float(np.quantile((calibration[columns] / scales).max(axis=1), 0.99))
    frozen = pd.read_csv(args.diagnostics / "lstm-scores.csv")
    frozen["t"] = pd.to_datetime(frozen.t, utc=True, format="mixed")
    candidate = frozen.copy()
    candidate["anomaly_score"] = (candidate[columns] / scales).max(axis=1)
    candidate["threshold"] = threshold
    candidate["is_alert"] = candidate.anomaly_score > threshold
    candidate["model_version"] = args.output.name
    diagnostics = json.loads((args.diagnostics / "diagnostics.json").read_text())
    baseline = diagnostics["models"]["lstm"]
    common = set(frozen.t)
    for name in ("if_30min", "if_original_raw"):
        other = pd.read_csv(args.diagnostics / f"{name}-scores.csv", usecols=["t"])
        common.intersection_update(pd.to_datetime(other.t, utc=True, format="mixed"))
    # Recover nominal membership from the exact packaged input, never inferred from alerts.
    import io
    import zipfile

    package = Path("artifacts/colab/hydropredict-lstm-input-2026-10-14.zip")
    if compute_file_sha256(package) != diagnostics["input_package_sha256"]:
        raise ValueError("Input package changed.")
    with zipfile.ZipFile(package) as archive:
        nominal = pd.read_csv(io.BytesIO(archive.read("data/validation_nominal.csv")))
    nominal_times = set(pd.to_datetime(nominal.t, utc=True, format="mixed"))
    faults = [pd.Timestamp(e["fault_time"]) for e in baseline["native_original_eligible"]["events"]]
    original_coverage = baseline["native_original_eligible"]["events"]
    matched = candidate[candidate.t.isin(common)]
    adequately_scored = event_metrics(matched, fault_window_coverage(matched.t, faults))
    normal_metrics = nominal_metrics(candidate[candidate.t.isin(common & nominal_times)])
    additional = (
        adequately_scored["detected_events"]
        > baseline["matched_adequately_scored"]["detected_events"]
    )
    result = {
        "channel_scales": scales.to_dict(),
        "threshold": threshold,
        "matched_original_eligible": event_metrics(matched, original_coverage),
        "matched_adequately_scored": adequately_scored,
        "matched_nominal": normal_metrics,
        "native_nominal": nominal_metrics(candidate[candidate.t.isin(nominal_times)]),
        "meets_development_criterion": additional
        and normal_metrics["point_alert_fraction"] <= 0.02,
        "promoted": False,
    }
    candidate.to_csv(args.output / "scores.csv", index=False)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    (args.output / "checksums.json").write_text(
        json.dumps(
            {p.name: compute_file_sha256(p) for p in args.output.iterdir() if p.is_file()}, indent=2
        )
    )
    print(
        json.dumps(
            {
                "threshold": threshold,
                "matched_nominal": normal_metrics,
                "detected": adequately_scored["detected_events"],
                "meets_development_criterion": result["meets_development_criterion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
