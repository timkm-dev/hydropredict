"""Loopback-only research dashboard and frozen LSTM CPU inference service."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import suppress
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from hydropredict.data.schema import MEASUREMENT_COLUMNS
from hydropredict.models.lstm import SHPLSTMAutoencoder

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "models/lstm-diagnostics-2026-10-15-v1/frozen_bundle_copy"
MAX_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100_000


class InferenceService:
    """Load one trusted local bundle; never accept client-selected model paths."""

    def __init__(self, bundle: Path) -> None:
        self.model = SHPLSTMAutoencoder.load(bundle)

    def info(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "model_version": self.model.version,
            "threshold": self.model.threshold,
            "sequence_length": config.sequence_length,
            "minimum_step_seconds": config.minimum_step_seconds,
            "maximum_step_seconds": config.maximum_step_seconds,
            "device": "cpu",
            "alert_rule": "score > threshold",
            "status": "Research model; not validated for operational warnings",
        }

    def score(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError("Send a JSON object containing a records array.")
        records = payload["records"]
        if not 1 <= len(records) <= MAX_ROWS:
            raise ValueError(f"Provide between 1 and {MAX_ROWS:,} observations.")
        validated = []
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict) or not isinstance(record.get("t"), str):
                raise ValueError(f"Observation {index}: an ISO timestamp t is required.")
            try:
                timestamp = datetime.fromisoformat(record["t"].replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"Observation {index}: invalid ISO timestamp.") from error
            row: dict[str, Any] = {"t": pd.to_datetime(timestamp, utc=True)}
            for channel in MEASUREMENT_COLUMNS:
                value = record.get(channel)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or abs(value) > 3.4e38
                ):
                    raise ValueError(f"Observation {index}: {channel} must be a finite number.")
                row[channel] = value
            validated.append(row)
        frame = pd.DataFrame(validated)
        scores = self.model.score(frame)
        # JSON-safe timestamps, native floats/bools, and finite values only.
        serialized = json.loads(scores.to_json(orient="records", date_format="iso"))
        if not scores.empty and scores.anomaly_score.isna().any():
            raise ValueError("Model produced invalid scores.")
        return {
            "model": self.info(),
            "input_observations": len(frame),
            "scored_observations": len(scores),
            "unscored_observations": len(frame) - len(scores),
            "alert_observations": int(scores.is_alert.sum()),
            "scores": serialized,
        }


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve only frontend assets; inference accepts same-origin JSON requests."""

    def __init__(self, *args: Any, service: InferenceService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, directory=str(ROOT / "frontend"), **kwargs)

    def respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/model":
            self.respond(200, self.service.info())
        elif self.path.startswith("/api/"):
            self.respond(404, {"error": "Unknown endpoint."})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/score":
            self.respond(404, {"error": "Unknown endpoint."})
            return
        port = self.server.server_port if isinstance(self.server, HTTPServer) else 0
        origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        origin = self.headers.get("Origin")
        if (origin and origin not in origins) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.respond(403, {"error": "Use the local dashboard to request inference."})
            return
        if self.headers.get_content_type() != "application/json":
            self.respond(415, {"error": "Content-Type must be application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BYTES:
                self.respond(413, {"error": "Request must be nonempty and at most 25 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            result = self.service.score(payload)
        except (ValueError, KeyError, OverflowError) as error:
            self.respond(400, {"error": str(error)})
            return
        except RuntimeError:
            self.respond(
                422, {"error": "Inference failed. Check telemetry and model compatibility."}
            )
            return
        self.respond(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        service = InferenceService(args.bundle)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Cannot load LSTM bundle: {error}\n")
    handler = partial(DashboardHandler, service=service)
    with HTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"HydroPredict: http://127.0.0.1:{args.port} · {service.model.version}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
