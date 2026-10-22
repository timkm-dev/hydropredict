"""Loopback-only research dashboard and frozen LSTM CPU inference service."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
from contextlib import suppress
from datetime import datetime
from functools import partial
from http.cookies import SimpleCookie
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import pandas as pd
import torch
from dotenv import load_dotenv

from hydropredict.auth import SESSION_SECONDS, AuthStore
from hydropredict.data.schema import MEASUREMENT_COLUMNS
from hydropredict.google_login import GoogleLogin
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

    def __init__(
        self,
        *args: Any,
        service: InferenceService,
        auth: AuthStore | None = None,
        google: GoogleLogin | None = None,
        **kwargs: Any,
    ) -> None:
        self.service = service
        self.auth = auth
        self.google = google
        self.session_cookie: str | None = None
        super().__init__(*args, directory=str(ROOT / "frontend"), **kwargs)

    def token(self, name: str = "hp_session") -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            return cookie[name].value if name in cookie else ""
        except Exception:
            return ""

    def respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.session_cookie is not None:
            self.send_header("Set-Cookie", self.session_cookie)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/auth/config":
            self.respond(200, {"google_enabled": self.google is not None})
            return
        if path in {"/api/auth/google/start", "/api/auth/google/callback"}:
            self.handle_google(path)
            return
        if self.auth is not None:
            user = self.auth.user(self.token())
            if self.path == "/api/auth/me":
                self.respond(200 if user else 401, {"user": user})
                return
            if self.path.startswith("/api/") and user is None:
                self.respond(401, {"error": "Please sign in to continue."})
                return
            if self.path in ("/", "/index.html") and (not user or not user["username"]):
                self.send_response(302)
                self.send_header("Location", "/auth.html")
                self.end_headers()
                return
        if self.path == "/api/model":
            self.respond(200, self.service.info())
        elif self.path.startswith("/api/"):
            self.respond(404, {"error": "Unknown endpoint."})
        else:
            super().do_GET()

    def redirect(self, location: str, cookies: tuple[str, ...] = ()) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        for cookie in cookies:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        # OAuth authorization codes must never appear in HTTP request logs.
        if self.path.startswith("/api/auth/google/"):
            self.log_message('"%s %s" %s %s', self.command, urlsplit(self.path).path, code, size)
        else:
            super().log_request(code, size)

    def handle_google(self, path: str) -> None:
        clear = "hp_oauth=; HttpOnly; SameSite=Lax; Path=/api/auth/google; Max-Age=0"
        try:
            if self.google is None or self.auth is None:
                raise ValueError("Google sign-in is not configured on this server.")
            query = parse_qs(urlsplit(self.path).query)
            if path.endswith("/start"):
                configured = urlsplit(self.google.redirect_uri)
                if self.headers.get("Host") != configured.netloc:
                    self.redirect(f"{configured.scheme}://{configured.netloc}{self.path}")
                    return
                link_user = None
                if query.get("link") == ["1"]:
                    user = self.auth.user(self.token())
                    if user is None:
                        raise ValueError("Sign in with your password before connecting Google.")
                    link_user = user["id"]
                state, nonce, verifier = self.auth.begin_google(link_user)
                self.redirect(
                    self.google.authorize(state, nonce, verifier),
                    (
                        f"hp_oauth={state}; HttpOnly; SameSite=Lax; "
                        "Path=/api/auth/google; Max-Age=600",
                    ),
                )
                return
            state = query.get("state", [""])[0]
            cookie = self.token("hp_oauth")
            if not state or not cookie or not secrets.compare_digest(state, cookie):
                raise ValueError("Google sign-in could not be matched to this browser. Try again.")
            flow = self.auth.consume_google(state)
            if "error" in query:
                raise ValueError(
                    "Google sign-in was cancelled. You can try again or use a password."
                )
            code = query.get("code", [""])[0]
            if not code or len(code) > 4096:
                raise ValueError("Google did not return a sign-in code. Please try again.")
            claims = self.google.exchange(code, flow["verifier"], flow["nonce"])
            token = self.auth.google_account(claims, flow["link_user"])
            self.auth.logout(self.token())
            user = self.auth.user(token)
            self.redirect(
                "/" if user and user["username"] else "/auth.html",
                (
                    clear,
                    f"hp_session={token}; HttpOnly; SameSite=Strict; "
                    f"Path=/; Max-Age={SESSION_SECONDS}",
                ),
            )
        except ValueError as error:
            self.redirect("/auth.html?" + urlencode({"error": str(error)}), (clear,))

    def do_POST(self) -> None:
        if self.path not in {
            "/api/score",
            "/api/auth/register",
            "/api/auth/login",
            "/api/auth/logout",
            "/api/auth/profile",
        }:
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
            if not isinstance(payload, dict):
                raise ValueError("Send a JSON object.")
            if self.path.startswith("/api/auth/"):
                self.handle_auth(payload)
                return
            if self.auth is not None:
                user = self.auth.user(self.token())
                if not user:
                    self.respond(401, {"error": "Please sign in to continue."})
                    return
                if not user["username"]:
                    self.respond(403, {"error": "Complete your profile first."})
                    return
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

    def handle_auth(self, payload: dict[str, Any]) -> None:
        if self.auth is None:
            self.respond(503, {"error": "Accounts are unavailable."})
            return
        action = self.path.rsplit("/", 1)[-1]
        token = self.token()
        if action in {"register", "login"}:
            token = (
                self.auth.register(payload) if action == "register" else self.auth.login(payload)
            )
            self.auth.logout(self.token())
            self.session_cookie = (
                f"hp_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}"
            )
        elif action == "logout":
            self.auth.logout(token)
            self.session_cookie = "hp_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
        elif action == "profile":
            user = self.auth.user(token)
            if user is None:
                self.respond(401, {"error": "Please sign in to continue."})
                return
            self.auth.profile(user["id"], payload)
        self.respond(200, {"user": self.auth.user(token)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--auth-db", type=Path, default=ROOT / "data/accounts.sqlite3")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        service = InferenceService(args.bundle)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Cannot load LSTM bundle: {error}\n")
    load_dotenv(ROOT / ".env", override=False)
    database_url = os.environ.get("DATABASE_URL")
    google = None
    if os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"):
        google = GoogleLogin(
            os.environ["GOOGLE_CLIENT_ID"],
            os.environ["GOOGLE_CLIENT_SECRET"],
            os.environ.get(
                "GOOGLE_REDIRECT_URI", f"http://127.0.0.1:{args.port}/api/auth/google/callback"
            ),
        )
    handler = partial(
        DashboardHandler,
        service=service,
        auth=AuthStore(database_url or args.auth_db),
        google=google,
    )
    with HTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"HydroPredict: http://127.0.0.1:{args.port} · {service.model.version}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
