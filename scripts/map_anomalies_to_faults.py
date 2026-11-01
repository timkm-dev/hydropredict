"""Map frozen anomaly decisions to fault windows without fitting or tuning."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from hydropredict.analysis.readiness import fault_window_coverage

ROOT = Path(__file__).resolve().parents[1]


def episodes(scores: pd.DataFrame) -> list[dict[str, Any]]:
    """An ordinary decision, segment change or >450s gap ends an episode."""
    result: list[dict[str, Any]] = []
    previous = None
    for row in scores.itertuples():
        gap = previous is None or (row.t - previous.t).total_seconds() > 450
        segment_change = previous is not None and getattr(row, "segment_id", None) != getattr(
            previous, "segment_id", None
        )
        if row.is_alert:
            if previous is None or not previous.is_alert or gap or segment_change:
                result.append({"start": row.t.isoformat(), "end": row.t.isoformat(), "points": 0})
            result[-1]["end"] = row.t.isoformat()
            result[-1]["points"] += 1
        previous = row
    return result


def map_events(scores: pd.DataFrame, faults: list[pd.Timestamp]) -> list[dict[str, Any]]:
    coverage = fault_window_coverage(scores.t, faults)
    result = []
    group = -1
    previous_fault = None
    for index, (fault, covered) in enumerate(zip(faults, coverage, strict=True)):
        start = fault - pd.Timedelta(hours=6)
        if previous_fault is None or start > previous_fault:
            group += 1
        previous_fault = fault
        window = scores[scores.t.ge(start) & scores.t.lt(fault)]
        alerts = window.loc[window.is_alert, "t"]
        status = (
            "insufficient" if not covered["eligible"] else "warned" if len(alerts) else "missed"
        )
        result.append(
            {
                "id": index + 1,
                "fault": fault.isoformat(),
                "group": group + 1,
                "status": status,
                "eligible": covered["eligible"],
                "coverage": covered["coverage_fraction"],
                "scored_points": len(window),
                "latest_gap_minutes": covered["last_observation_gap_seconds"] / 60
                if covered["last_observation_gap_seconds"] is not None
                else None,
                "alert_points": len(alerts),
                "first_alert": alerts.min().isoformat() if len(alerts) else None,
                "lead_minutes": (fault - alerts.min()).total_seconds() / 60
                if len(alerts)
                else None,
                "peak_ratio": float(window.anomaly_score.max() / window.threshold.iloc[0])
                if len(window)
                else None,
            }
        )
    return result


def generate(inputs: Path, diagnostics: Path, output: Path) -> dict[str, Any]:
    raw = inputs.read_bytes()
    diagnostic = json.loads((diagnostics / "diagnostics.json").read_text())
    if hashlib.sha256(raw).hexdigest().upper() != diagnostic["input_package_sha256"].upper():
        raise ValueError("Input package differs from frozen evidence")
    checks = json.loads((diagnostics / "checksums.json").read_text())
    for name in ("diagnostics.json", "lstm-scores.csv"):
        if (
            hashlib.sha256((diagnostics / name).read_bytes()).hexdigest().upper()
            != checks[name].upper()
        ):
            raise ValueError(f"Changed diagnostic evidence: {name}")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        manifest = json.loads(archive.read("input_manifest.json"))
        if manifest["test_data_included"]:
            raise ValueError("Only development data is allowed")
        for name, expected in manifest["files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest().upper() != expected.upper():
                raise ValueError(f"Changed package member: {name}")

        def read(name: str) -> pd.DataFrame:
            frame = pd.read_csv(io.BytesIO(archive.read(f"data/{name}.csv")))
            frame["t"] = pd.to_datetime(frame.t, utc=True, format="mixed")
            return frame

        faults = sorted(read("evaluation_faults").t.tolist())
        nominal = read("validation_nominal")
        timeline = read("validation_context")
    scores = pd.read_csv(diagnostics / "lstm-scores.csv")
    scores["t"] = pd.to_datetime(scores.t, utc=True, format="mixed")
    lower, upper = map(pd.Timestamp, manifest["validation_interval"])
    if (
        scores.t.isna().any()
        or scores.t.duplicated().any()
        or not scores.t.is_monotonic_increasing
        or not scores.t.between(lower, upper, inclusive="left").all()
        or not all(lower <= f < upper for f in faults)
    ):
        raise ValueError("Invalid development timestamps")
    threshold = float(scores.threshold.iloc[0])
    if threshold <= 0 or not scores.threshold.eq(threshold).all():
        raise ValueError("Expected one frozen positive threshold")
    if not scores.is_alert.equals(scores.anomaly_score.gt(threshold)):
        raise ValueError("Saved decisions disagree with frozen strict threshold")
    events = map_events(scores, faults)
    alert_episodes = episodes(scores)
    alert_times = scores.loc[scores.is_alert, "t"]
    associated = pd.Series(False, index=alert_times.index)
    for fault in faults:
        associated |= alert_times.ge(fault - pd.Timedelta(hours=6)) & alert_times.lt(fault)
    selected_nominal = scores[scores.t.isin(nominal.t)]
    payload = {
        "schema_version": 1,
        "model": str(scores.model_version.iloc[0]),
        "threshold": threshold,
        "period": [lower.isoformat(), upper.isoformat()],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "scores_sha256": checks["lstm-scores.csv"],
            "source_commit": diagnostic["source_commit"],
        },
        "summary": {
            "faults": len(events),
            "eligible": sum(e["eligible"] for e in events),
            "warned": sum(e["status"] == "warned" for e in events),
            "missed": sum(e["status"] == "missed" for e in events),
            "insufficient": sum(e["status"] == "insufficient" for e in events),
            "score_points": len(scores),
            "observations": len(timeline),
            "alert_points": len(alert_times),
            "episodes": len(alert_episodes),
            "associated_alert_points": int(associated.sum()),
            "outside_alert_points": int((~associated).sum()),
            "nominal_points": len(selected_nominal),
            "nominal_alert_points": int(selected_nominal.is_alert.sum()),
        },
        "events": events,
        "episodes": alert_episodes,
        "scores": [
            {
                "t": row.t.isoformat(),
                "ratio": float(row.anomaly_score / threshold),
                "alert": bool(row.is_alert),
            }
            for row in scores.itertuples()
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "artifacts/colab/hydropredict-lstm-input-2026-10-14.zip",
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=ROOT / "models/lstm-diagnostics-2026-10-15-v1"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "frontend/evidence-data.json")
    args = parser.parse_args()
    print(json.dumps(generate(args.inputs, args.diagnostics, args.output)["summary"], indent=2))
