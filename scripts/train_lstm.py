"""Run the declared LSTM experiment on Colab, or a clearly labelled CPU smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from hydropredict.data.partitioning import compute_file_sha256
from hydropredict.data.preprocessing import load_telemetry_dataframe
from hydropredict.models.lstm import LSTMConfig, train_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/lstm_initial.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    manifest_path = args.data_dir.parent / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_path = args.data_dir / "normal_training.csv"
    expected = manifest["files_sha256"]["data/normal_training.csv"]
    if compute_file_sha256(training_path) != expected:
        raise ValueError("Training input checksum mismatch.")
    config = LSTMConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    normal = load_telemetry_dataframe(training_path)
    if normal.t.max() >= pd.Timestamp(manifest["validation_interval"][0]):
        raise ValueError("Normal training must finish before development validation starts.")
    fit_end = int(len(normal) * 0.8)
    stopping_start = int(fit_end * 0.9)
    fitting = normal.iloc[:stopping_start].copy()
    stopping = normal.iloc[stopping_start:fit_end].copy()
    calibration = normal.iloc[fit_end:].copy()
    if args.smoke_test:
        torch.set_num_threads(1)
        fitting, stopping, calibration = (
            fitting.iloc[:256],
            stopping.iloc[:96],
            calibration.iloc[:96],
        )
        config = replace(config, maximum_epochs=2, patience=2)
    provenance = {
        "source_commit": manifest["source_commit"],
        "training_input_hash": expected,
        "status": "CPU engineering smoke test only" if args.smoke_test else "Colab GPU experiment",
        "fit_observations": len(fitting),
        "early_stopping_observations": len(stopping),
        "calibration_observations": len(calibration),
        "faults_used_for_training": False,
    }
    train_lstm(
        fitting,
        stopping,
        calibration,
        args.output,
        config,
        provenance,
        device="cpu" if args.smoke_test else "cuda",
        resume=args.resume,
    )
    print(f"Bundle exported to {args.output / 'bundle'}. Fault-based evaluation remains separate.")


if __name__ == "__main__":
    main()
