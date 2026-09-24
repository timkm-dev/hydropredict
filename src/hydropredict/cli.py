"""Command-line interface for HydroPredict development workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from hydropredict.data.validation import (
    SchemaError,
    validate_csv,
    write_accepted,
    write_quarantine,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hydropredict")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a telemetry CSV file")
    validate.add_argument("input", help="path to the input CSV file")
    validate.add_argument("--accepted-output", help="optional path for accepted records")
    validate.add_argument("--quarantine-output", help="optional path for rejected records")
    validate.add_argument(
        "--fail-on-rejected",
        action="store_true",
        help="return a non-zero status when any row is rejected",
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
