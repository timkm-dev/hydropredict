"""Evaluate an exported LSTM on declared validation inputs, without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from compare_validation_models import event_metrics, nominal_metrics

from hydropredict.analysis.readiness import fault_window_coverage
from hydropredict.data.faults import load_fault_log
from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.models.lstm import SHPLSTMAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior evaluations; choose a new output directory.")
    torch.set_num_threads(1)
    model = SHPLSTMAutoencoder.load(args.bundle)
    metadata = json.loads((args.bundle / "metadata.json").read_text(encoding="utf-8"))
    if metadata["provenance"]["provenance"].get("status") == "CPU engineering smoke test only":
        raise ValueError(
            "Smoke-test bundles are not scientific experiments; run full Colab training."
        )
    manifest = json.loads(
        (args.data_dir.parent / "input_manifest.json").read_text(encoding="utf-8")
    )
    training_provenance = metadata["provenance"]["provenance"]
    if (
        training_provenance["source_commit"] != manifest["source_commit"]
        or training_provenance["training_input_hash"]
        != manifest["files_sha256"]["data/normal_training.csv"]
    ):
        raise ValueError("Evaluation inputs do not match the model's source snapshot.")
    names = (
        "validation_context.csv",
        "validation_prefault.csv",
        "validation_nominal.csv",
        "evaluation_faults.csv",
    )
    for name in names:
        if compute_file_sha256(args.data_dir / name) != manifest["files_sha256"][f"data/{name}"]:
            raise ValueError(f"Evaluation file changed: {name}")
    frames = {name: load_telemetry_dataframe(args.data_dir / name) for name in names[:-1]}
    lower, upper = [pd.Timestamp(t) for t in manifest["validation_interval"]]
    if pd.Timestamp(metadata["provenance"]["partition_ranges"]["calibration"]["end"]) >= lower:
        raise ValueError("Calibration must precede development validation.")
    timeline = frames["validation_context.csv"]
    if not timeline.t.between(lower, upper, inclusive="left").all():
        raise ValueError("Evaluation observations must remain within validation interval.")
    scores = model.score(timeline)
    faults = [
        f for f in load_fault_log(args.data_dir / "evaluation_faults.csv") if lower <= f < upper
    ]
    original = fault_window_coverage(frames["validation_prefault.csv"].t, faults)
    effective = fault_window_coverage(scores.t, faults)
    nominal = scores[scores.t.isin(frames["validation_nominal.csv"].t)]
    report = {
        "status": "Development validation; not independent final test",
        "original_eligible": event_metrics(scores, original),
        "adequately_scored": event_metrics(scores, effective),
        "nominal": nominal_metrics(nominal) if len(nominal) else None,
        "sequence_accounting": model.sequences(timeline).accounting,
        "nominal_caveat": "Selected normal reference; not fleet false-alert rate",
        "test_evaluated": False,
    }
    args.output.mkdir(parents=True)
    scores.to_csv(args.output / "scores.csv", index=False)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("original_eligible", "adequately_scored")
            },
            indent=2,
        )
    )
    print(
        "Adequately scored event results:",
        {k: v for k, v in report["adequately_scored"].items() if k != "events"},
    )


if __name__ == "__main__":
    main()
