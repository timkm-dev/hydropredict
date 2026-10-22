"""OAuth identity, browser binding, token validation and linking regression tests."""

from __future__ import annotations

import http.client
import threading
import time
from functools import partial
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from hydropredict.auth import AuthStore
from hydropredict.dashboard import DashboardHandler
from hydropredict.google_login import GoogleLogin


def claims(subject: str = "google-123", email: str = "google@example.com") -> dict[str, Any]:
    return {"sub": subject, "email": email, "email_verified": True, "name": "Google User"}


def test_google_identity_and_explicit_linking(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.sqlite3")
    token = store.google_account(claims())
    user = store.user(token)
    assert user and user["name"] == "Google User" and user["google_connected"]
    assert user["username"] is None
    # Google's stable subject identifies the account even after an email change.
    again = store.user(store.google_account(claims(email="changed@example.com")))
    assert again and again["id"] == user["id"]
    with pytest.raises(ValueError, match="Incorrect"):
        store.login({"email": "google@example.com", "password": "anything"})
    credentials = {"email": "existing@example.com", "password": "a valid long password"}
    local = store.user(store.register(credentials))
    assert local
    with pytest.raises(ValueError, match="already uses"):
        store.google_account(claims("another-google", credentials["email"]))
    linked = store.user(
        store.google_account(claims("another-google", credentials["email"]), local["id"])
    )
    assert linked and linked["id"] == local["id"] and linked["google_connected"]
    assert store.user(store.login(credentials))
    with pytest.raises(ValueError, match="another account"):
        store.google_account(claims(), local["id"])
    with pytest.raises(ValueError, match="verified"):
        store.google_account({**claims(), "email_verified": False})


def test_oauth_state_expires_and_cannot_be_reused(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.sqlite3")
    state, nonce, verifier = store.begin_google()
    flow = store.consume_google(state)
    assert flow["nonce"] == nonce and flow["verifier"] == verifier
    with pytest.raises(ValueError, match="expired"):
        store.consume_google(state)
    state, _, _ = store.begin_google()
    with store.connect() as db:
        db.execute("UPDATE oauth_states SET expires=?", (time.time() - 1,))
    with pytest.raises(ValueError, match="expired"):
        store.consume_google(state)


def test_exchange_checks_nonce_and_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    google = GoogleLogin(
        "test-client", "private-secret", "http://127.0.0.1:8766/api/auth/google/callback"
    )
    calls: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"id_token": "signed-token"}

    def post(url: str, **kwargs: Any) -> Response:
        calls.update(kwargs)
        return Response()

    def verify(token: str, request: Any, audience: str) -> dict[str, Any]:
        assert token == "signed-token" and audience == "test-client"
        return {**claims(), "nonce": "expected-nonce"}

    monkeypatch.setattr("hydropredict.google_login.requests.post", post)
    monkeypatch.setattr("hydropredict.google_login.id_token.verify_oauth2_token", verify)
    assert google.exchange("code", "verifier", "expected-nonce")["sub"] == "google-123"
    assert calls["data"]["code_verifier"] == "verifier"
    assert calls["timeout"] == 15
    with pytest.raises(ValueError, match="could not be verified"):
        google.exchange("code", "verifier", "wrong-nonce")
    with pytest.raises(ValueError, match="local dashboard"):
        GoogleLogin("id", "secret", "http://untrusted.example/api/auth/google/callback")


def test_http_google_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auth = AuthStore(tmp_path / "auth.sqlite3")
    with HTTPServer(("127.0.0.1", 0), DashboardHandler) as server:
        google = GoogleLogin(
            "test-client",
            "private-secret",
            f"http://127.0.0.1:{server.server_port}/api/auth/google/callback",
        )
        server.RequestHandlerClass = partial(
            DashboardHandler, service=None, auth=auth, google=google
        )
        monkeypatch.setattr(GoogleLogin, "exchange", lambda *args: claims())
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET", "/api/auth/google/start?link=1")
            response = connection.getresponse()
            assert "Sign+in" in (response.getheader("Location") or "")
            response.read()
            connection.request("GET", "/api/auth/google/start")
            response = connection.getresponse()
            location = response.getheader("Location") or ""
            state = parse_qs(urlsplit(location).query)["state"][0]
            assert "private-secret" not in location and "code_challenge_method=S256" in location
            cookie = (response.getheader("Set-Cookie") or "").split(";", 1)[0]
            assert "SameSite=Lax" in (response.getheader("Set-Cookie") or "")
            response.read()
            callback = f"/api/auth/google/callback?code=test-code&state={state}"
            connection.request("GET", callback)
            response = connection.getresponse()
            assert "could+not+be+matched" in (response.getheader("Location") or "")
            response.read()
            connection.request("GET", callback, headers={"Cookie": cookie})
            response = connection.getresponse()
            assert response.getheader("Location") == "/auth.html"
            cookies = [value for key, value in response.getheaders() if key.lower() == "set-cookie"]
            session = next(
                value.split(";", 1)[0] for value in cookies if value.startswith("hp_session=")
            )
            response.read()
            connection.request("GET", "/api/auth/me", headers={"Cookie": session})
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.request("GET", callback, headers={"Cookie": cookie})
            response = connection.getresponse()
            assert "expired" in (response.getheader("Location") or "")
            response.read()
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)
