"""Command-line interface for HydroPredict development workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hydropredict.analysis.eda import compute_health_distribution_profile, format_eda_summary
from hydropredict.data.faults import (
    FaultAlignmentConfig,
    align_telemetry_file,
)
from hydropredict.data.partitioning import create_shp_partitions
from hydropredict.data.preprocessing import (
    RobustTelemetryScaler,
    StandardTelemetryScaler,
    assess_sampling_cadence,
    load_scaler,
    load_telemetry_dataframe,
    prepare_telemetry_series,
)
from hydropredict.data.validation import (
    SchemaError,
    validate_csv,
    write_accepted,
    write_quarantine,
)
from hydropredict.features.pipeline import FeatureConfig, extract_feature_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hydropredict")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. validate
    validate = subparsers.add_parser("validate", help="validate a telemetry CSV file")
    validate.add_argument("input", help="path to the input CSV file")
    validate.add_argument("--accepted-output", help="optional path for accepted records")
    validate.add_argument("--quarantine-output", help="optional path for rejected records")
    validate.add_argument(
        "--fail-on-rejected",
        action="store_true",
        help="return a non-zero status when any row is rejected",
    )

    # 2. preprocess
    preprocess = subparsers.add_parser(
        "preprocess", help="resample onto regular grid, isolate segments, and scale telemetry"
    )
    preprocess.add_argument("input", help="path to validated telemetry CSV")
    preprocess.add_argument("--output", help="optional output path for preprocessed telemetry CSV")
    preprocess.add_argument(
        "--fit-scaler", help="optional output path to fit and save a new scaler JSON"
    )
    preprocess.add_argument(
        "--scaler-output", help="alias for --fit-scaler for backwards compatibility"
    )
    preprocess.add_argument(
        "--scaler", help="path to an existing frozen scaler JSON to load and apply"
    )
    preprocess.add_argument(
        "--scale", action="store_true", help="apply fitted or loaded scaler to exported telemetry"
    )
    preprocess.add_argument(
        "--scaler-type",
        choices=["robust", "standard"],
        default="robust",
        help="scaler type to fit: robust (median/IQR) or standard (mean/std) (default: robust)",
    )
    preprocess.add_argument(
        "--freq", default="5min", help="regular temporal frequency (default: 5min)"
    )
    preprocess.add_argument(
        "--method",
        choices=["forward", "linear"],
        default="forward",
        help="gap filling method: forward (causal) or linear (default: forward)",
    )
    preprocess.add_argument(
        "--accounting-output", help="optional path to export row accounting JSON report"
    )

    # 3. extract-features
    features = subparsers.add_parser(
        "extract-features", help="compute rolling, transient, and domain features"
    )
    features.add_argument("input", help="path to preprocessed telemetry CSV")
    features.add_argument("--output", required=True, help="output path for feature matrix CSV")
    features.add_argument(
        "--windows",
        default="6,24",
        help="comma-separated rolling window sizes (default: 6,24)",
    )
    features.add_argument("--no-transients", action="store_true", help="exclude transient features")
    features.add_argument(
        "--no-domain", action="store_true", help="exclude domain physics features"
    )

    # 4. align-faults
    align = subparsers.add_parser(
        "align-faults", help="align telemetry with ground-truth fault log"
    )
    align.add_argument("input", help="path to telemetry CSV")
    align.add_argument("faults", help="path to ground-truth faults CSV")
    align.add_argument("--output", required=True, help="output path for labeled telemetry CSV")
    align.add_argument(
        "--precursor-hours",
        type=float,
        default=6.0,
        help="pre-fault warning window in hours (default: 6.0)",
    )
    align.add_argument(
        "--fault-duration-mins",
        type=float,
        default=30.0,
        help="active fault window in minutes (default: 30.0)",
    )

    # 5. eda
    eda = subparsers.add_parser("eda", help="statistical EDA comparing normal vs precursor states")
    eda.add_argument("input", help="path to labeled telemetry CSV (with fault_state column)")
    eda.add_argument(
        "--json", action="store_true", help="output JSON summary instead of text table"
    )

    # 6. partition
    partition_parser = subparsers.add_parser(
        "partition", help="partition SHP telemetry into chronological or published benchmark sets"
    )
    partition_parser.add_argument(
        "--train-source", default="data/raw/shp_train.csv", help="path to raw training telemetry"
    )
    partition_parser.add_argument(
        "--test-source", default="data/raw/shp_test.csv", help="path to raw test telemetry"
    )
    partition_parser.add_argument(
        "--faults-source", default="data/raw/shp_faults.csv", help="path to raw fault log"
    )
    partition_parser.add_argument(
        "--strategy",
        choices=["chronological", "published"],
        default="chronological",
        help="partition strategy (default: chronological)",
    )
    partition_parser.add_argument(
        "--output-dir",
        default="data/processed/partitions",
        help="output directory for partition files and manifest",
    )
    partition_parser.add_argument(
        "--train-end",
        default="2019-02-01T00:00:00Z",
        help="cutoff timestamp ending training partition in chronological mode",
    )
    partition_parser.add_argument(
        "--val-end",
        default="2019-05-01T00:00:00Z",
        help="cutoff timestamp ending validation partition in chronological mode",
    )

    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)

    if args.command == "validate":
        try:
            report = validate_csv(args.input)
        except (OSError, SchemaError) as error:
            print(json.dumps({"status": "error", "message": str(error)}, indent=2))
            return 2

        if args.accepted_output:
            write_accepted(report, args.accepted_output)
        if args.quarantine_output:
            write_quarantine(report, args.quarantine_output)

        print(json.dumps({"status": "complete", **report.summary()}, indent=2))
        return 1 if args.fail_on_rejected and report.rejected_count else 0

    if args.command == "preprocess":
        try:
            df = load_telemetry_dataframe(args.input)
            cadence = assess_sampling_cadence(df).summary() if len(df) >= 2 else None
            prepared, accounting = prepare_telemetry_series(
                df,
                freq=args.freq,
                method=args.method,
            )

            # Resolve scaler fitting or loading
            scaler: RobustTelemetryScaler | StandardTelemetryScaler | None = None
            fit_path = args.fit_scaler or args.scaler_output
            if fit_path:
                if args.scaler_type == "standard":
                    scaler = StandardTelemetryScaler().fit(prepared)
                else:
                    scaler = RobustTelemetryScaler().fit(prepared)
                scaler.save(fit_path)
            elif args.scaler:
                scaler = load_scaler(args.scaler)

            export_df = prepared
            if args.scale:
                if scaler is None:
                    raise ValueError(
                        "Cannot apply scaling without a scaler. Specify --fit-scaler or --scaler."
                    )
                export_df = scaler.transform(prepared)

            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                export_df.to_csv(out_path, index=False)

            if args.accounting_output:
                acc_path = Path(args.accounting_output)
                acc_path.parent.mkdir(parents=True, exist_ok=True)
                with acc_path.open("w", encoding="utf-8") as handle:
                    json.dump(accounting.summary(), handle, indent=2)

            result_summary = {
                "status": "complete",
                "original_records": len(df),
                "prepared_records": len(prepared),
                "is_scaled": bool(args.scale),
                "segments": accounting.segment_count,
                "imputed_steps": accounting.imputed_step_count,
                "cadence": cadence,
            }
            print(json.dumps(result_summary, indent=2))
            return 0
        except Exception as error:
            print(json.dumps({"status": "error", "message": str(error)}, indent=2))
            return 2

    if args.command == "extract-features":
        try:
            df = pd.read_csv(args.input)
            windows_tuple = tuple(int(w.strip()) for w in args.windows.split(",") if w.strip())
            cfg = FeatureConfig(
                include_raw=True,
                rolling_windows=windows_tuple,
                include_transients=not args.no_transients,
                include_domain=not args.no_domain,
            )
            features_df = extract_feature_matrix(df, config=cfg)

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(out_path, index=False)

            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output_file": str(out_path),
                        "total_rows": len(features_df),
                        "feature_columns": len(features_df.columns),
                    },
                    indent=2,
                )
            )
            return 0
        except Exception as error:
            print(json.dumps({"status": "error", "message": str(error)}, indent=2))
            return 2

    if args.command == "align-faults":
        try:
            fault_cfg = FaultAlignmentConfig(
                precursor_window_seconds=args.precursor_hours * 3600.0,
                fault_duration_seconds=args.fault_duration_mins * 60.0,
            )
            aligned = align_telemetry_file(
                args.input, args.faults, output_csv=args.output, config=fault_cfg
            )
            counts = aligned["fault_state"].value_counts().to_dict()
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "total_rows": len(aligned),
                        "normal_rows": counts.get(0, 0),
                        "precursor_rows": counts.get(1, 0),
                        "fault_rows": counts.get(2, 0),
                    },
                    indent=2,
                )
            )
            return 0
        except Exception as error:
            print(json.dumps({"status": "error", "message": str(error)}, indent=2))
            return 2

    if args.command == "eda":
        try:
            df = pd.read_csv(args.input)
            profile = compute_health_distribution_profile(df)
            if args.json:
                print(json.dumps(profile, indent=2))
            else:
                print(format_eda_summary(profile))
            return 0
        except Exception as error:
            print(json.dumps({"status": "error", "message": str(error)}, indent=2))
            return 2

    if args.command == "partition":
        try:
            manifest = create_shp_partitions(
                train_source=args.train_source,
                test_source=args.test_source,
                faults_source=args.faults_source,
                output_dir=args.output_dir,
                strategy=args.strategy,
                train_end_boundary=args.train_end,
                val_end_boundary=args.val_end,
            )
            print(json.dumps({"status": "complete", **manifest.to_dict()}, indent=2))
            return 0
        except Exception as error:
            print(json.dumps({"status": "error", "message": str(error)}, indent=2))
            return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
