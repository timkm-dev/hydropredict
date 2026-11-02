"""Exercise real frozen-network scoring and the local HTTP boundary."""

from __future__ import annotations

import http.client
import json
import threading
from functools import partial
from http.server import HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from hydropredict.assessment_store import AssessmentStore
from hydropredict.auth import AuthStore
from hydropredict.dashboard import DashboardHandler, InferenceService
from hydropredict.models.lstm import LSTMConfig, SHPLSTMAutoencoder


@pytest.fixture
def service(tmp_path: Path) -> InferenceService:
    frame = pd.DataFrame(np.arange(120).reshape(20, 6), columns=[f"V{i}" for i in range(1, 7)])
    model = SHPLSTMAutoencoder(LSTMConfig(hidden_size=4, latent_size=2))
    model.scaler.fit(frame)
    model.threshold = 0.3
    model.version = "synthetic-test"
    model.save(tmp_path / "bundle", {})
    return InferenceService(tmp_path / "bundle")


def records() -> list[dict[str, Any]]:
    return [
        {"t": t.isoformat(), **{f"V{i}": index + i for i in range(1, 7)}}
        for index, t in enumerate(pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC"))
    ]


def test_saved_assessment_http_flow(service: InferenceService, tmp_path: Path) -> None:
    auth = AuthStore(tmp_path / "accounts.sqlite3", ["review@example.com"])
    token = auth.register({"email": "review@example.com", "password": "long test password"})
    user = auth.user(token)
    assert user is not None
    auth.profile(
        user["id"],
        {
            "name": "Reviewer",
            "company": "Test",
            "username": "reviewer",
            "job_position": "Engineer",
        },
    )
    store = AssessmentStore(tmp_path / "assessments.sqlite3")
    handler = partial(DashboardHandler, service=service, auth=auth, assessments=store)
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        headers = {"Content-Type": "application/json", "Cookie": f"hp_session={token}"}
        try:
            connection.request(
                "POST",
                "/api/assess",
                json.dumps({"name": "test.csv", "records": records()}),
                headers,
            )
            response = connection.getresponse()
            assert response.status == 200
            run = json.loads(response.read())
            assert run["result"]["scored_observations"] == 14
            connection.request("GET", "/api/run?id=" + run["id"], headers=headers)
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["records"] == records()
            connection.request("GET", "/", headers=headers)
            response = connection.getresponse()
            assert response.status == 200
            assert b"care.js" in response.read()
            connection.request("GET", "/shp.html", headers=headers)
            response = connection.getresponse()
            assert response.status == 200
            assert b"Review unusual behaviour" in response.read()
            connection.request("GET", "/api/runs")
            response = connection.getresponse()
            assert response.status == 401
            response.read()
            connection.request(
                "POST",
                "/api/review",
                json.dumps({"run_id": run["id"], "episode": "invalid", "action": "acknowledged"}),
                headers,
            )
            response = connection.getresponse()
            assert response.status == 400
            response.read()
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)


def test_scores_match_direct_inference_and_account_for_gaps(service: InferenceService) -> None:
    rows = records()
    result = service.score({"records": rows})
    expected = service.model.score(pd.DataFrame(rows))
    assert result["scored_observations"] == 14
    assert result["unscored_observations"] == 6
    np.testing.assert_allclose(
        [row["anomaly_score"] for row in result["scores"]], expected.anomaly_score
    )
    assert [row["is_alert"] for row in result["scores"]] == expected.is_alert.tolist()
    assert "V1_reconstruction_error" in result["scores"][0]
    assert service.score({"records": rows[:3]})["scores"] == []
    # Two segments of 10 readings each lose six warm-up observations each.
    for row in rows[10:]:
        row["t"] = (pd.Timestamp(row["t"]) + pd.Timedelta(hours=2)).isoformat()
    assert service.score({"records": rows})["unscored_observations"] == 12


@pytest.mark.parametrize("value", [None, True, "2", float("nan"), float("inf")])
def test_rejects_invalid_measurements(service: InferenceService, value: Any) -> None:
    rows = records()
    rows[0]["V1"] = value
    with pytest.raises(ValueError, match="finite number"):
        service.score({"records": rows})


def test_rejects_duplicate_and_unordered_times(service: InferenceService) -> None:
    with pytest.raises(ValueError, match="chronological"):
        service.score({"records": records()[::-1]})
    with pytest.raises(ValueError, match="unique"):
        service.score({"records": [records()[0]] * 7})


def test_http_boundary_and_model_status(service: InferenceService) -> None:
    with HTTPServer(("127.0.0.1", 0), partial(DashboardHandler, service=service)) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET", "/api/model")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["model_version"] == "synthetic-test"
            for headers, body, status in [
                ({"Content-Type": "application/json"}, json.dumps({"records": records()}), 200),
                ({"Content-Type": "application/json"}, "{}", 400),
                ({"Content-Type": "text/plain"}, "{}", 415),
                ({"Content-Type": "application/json", "Origin": "https://example.org"}, "{}", 403),
            ]:
                connection.request("POST", "/api/score", body, headers)
                response = connection.getresponse()
                assert response.status == status
                response.read()
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)


def test_hosted_origin_and_secure_cookie(service: InferenceService, tmp_path: Path) -> None:
    origin = "https://hydropredict.example.net"
    auth = AuthStore(tmp_path / "accounts.sqlite3")
    handler = partial(DashboardHandler, service=service, auth=auth, public_origin=origin)
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        credentials = json.dumps({"email": "host@example.com", "password": "a long test password"})
        try:
            for headers_origin, status in [(origin, 200), ("https://example.org", 403)]:
                headers = {"Content-Type": "application/json", "Origin": headers_origin}
                connection.request("POST", "/api/auth/register", credentials, headers)
                response = connection.getresponse()
                assert response.status == status
                if status == 200:
                    cookie = response.getheader("Set-Cookie")
                    assert cookie and cookie.endswith("; Secure")
                response.read()
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)


def test_authenticated_http_flow(service: InferenceService, tmp_path: Path) -> None:
    auth = AuthStore(tmp_path / "accounts.sqlite3", ["Test@Example.com "])
    handler = partial(DashboardHandler, service=service, auth=auth)
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET", "/api/model")
            response = connection.getresponse()
            assert response.status == 401
            response.read()
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert b"welcome-illustration.svg" in response.read()
            connection.request("GET", "/care.html")
            response = connection.getresponse()
            assert response.status == 302
            assert response.getheader("Location") == "/auth.html"
            response.read()
            headers = {"Content-Type": "application/json"}
            credentials = {"email": "test@example.com", "password": "a long test password"}
            connection.request("POST", "/api/auth/register", json.dumps(credentials), headers)
            response = connection.getresponse()
            assert response.status == 200
            cookie = response.getheader("Set-Cookie")
            assert cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
            assert json.loads(response.read())["user"]["username"] is None
            headers["Cookie"] = cookie.split(";", 1)[0]
            connection.request("POST", "/api/score", json.dumps({"records": records()}), headers)
            response = connection.getresponse()
            assert response.status == 403
            response.read()
            profile = {
                "name": "Test Person",
                "company": "Hydro",
                "username": "tester",
                "job_position": "Engineer",
            }
            connection.request("POST", "/api/auth/profile", json.dumps(profile), headers)
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["user"]["name"] == "Test Person"
            connection.request("POST", "/api/score", json.dumps({"records": records()}), headers)
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.request("POST", "/api/auth/logout", "{}", headers)
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.request("GET", "/api/model", headers=headers)
            response = connection.getresponse()
            assert response.status == 401
            response.read()
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)


def test_care_replay_data_and_alert_reviews(service: InferenceService, tmp_path: Path) -> None:
    care = tmp_path / "care-data"
    care.mkdir()
    fleet = {"periods": [{"key": "C-47", "alerts": [{"id": "C-47-1"}]}]}
    (care / "fleet.json").write_text(json.dumps(fleet))
    handler = partial(
        DashboardHandler,
        service=service,
        auth=AuthStore(tmp_path / "accounts.sqlite3"),
        assessments=AssessmentStore(tmp_path / "assessments.sqlite3"),
        care_data=care,
    )
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)

        def call(method: str, path: str, body: Any = None, headers: Any = None) -> Any:
            data = None if body is None else json.dumps(body)
            connection.request(method, path, data, headers or {})
            response = connection.getresponse()
            return response.status, response.getheader("Set-Cookie"), response.read()

        try:
            assert call("GET", "/care-data/fleet.json")[0] == 401
            assert call("GET", "/care.html")[0] == 302
            headers = {"Content-Type": "application/json"}
            credentials = {"email": "care@example.com", "password": "a long test password"}
            _, cookie, _ = call("POST", "/api/auth/register", credentials, headers)
            headers["Cookie"] = cookie.split(";", 1)[0]
            profile = {"name": "T", "company": "W", "username": "tech", "job_position": "Tech"}
            assert call("POST", "/api/auth/profile", profile, headers)[0] == 200
            status, _, body = call("GET", "/", headers=headers)
            assert status == 200 and b"care.js" in body
            review = {"alert": "C-47-1", "action": "inspect", "comment": "Check pump A"}
            status, _, body = call("POST", "/api/care/review", review, headers)
            assert status == 200 and json.loads(body)["reviews"][0]["action"] == "inspect"
            missing = {"alert": "C-47-1", "action": "dismissed"}
            assert call("POST", "/api/care/review", missing, headers)[0] == 400
            unknown = {"alert": "Z-1-1", "action": "inspect"}
            assert call("POST", "/api/care/review", unknown, headers)[0] == 400
            status, _, body = call("GET", "/api/care/reviews", headers=headers)
            assert [r["alert"] for r in json.loads(body)["reviews"]] == ["C-47-1"]
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)


def test_role_permissions(service: InferenceService, tmp_path: Path) -> None:
    auth = AuthStore(tmp_path / "accounts.sqlite3", ["ml@example.com"])
    handler = partial(
        DashboardHandler,
        service=service,
        auth=auth,
        assessments=AssessmentStore(tmp_path / "assessments.sqlite3"),
    )
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)

        def call(method: str, path: str, body: Any = None, cookie: str = "") -> Any:
            headers = {"Content-Type": "application/json", "Cookie": cookie}
            connection.request(method, path, None if body is None else json.dumps(body), headers)
            response = connection.getresponse()
            return response.status, response.getheader("Set-Cookie"), response.read()

        def account(email: str, username: str) -> str:
            credentials = {"email": email, "password": "a long test password"}
            cookie = call("POST", "/api/auth/register", credentials)[1].split(";", 1)[0]
            profile = {"name": "N", "company": "C", "username": username, "job_position": "ML"}
            assert call("POST", "/api/auth/profile", profile, cookie)[0] == 200
            return cookie

        try:
            operator = account("operator@example.com", "operator")
            engineer = account("ml@example.com", "engineer")
            me = json.loads(call("GET", "/api/auth/me", cookie=operator)[2])["user"]
            # A self-entered job title of "ML" does not grant the role.
            assert me["role"] == "operator"
            assert json.loads(call("GET", "/api/auth/me", cookie=engineer)[2])["user"]["role"] == (
                "ml_engineer"
            )
            for path in ["/care.html", "/api/care/reviews"]:
                assert call("GET", path, cookie=operator)[0] == 200
            for path in ["/models.html", "/shp.html", "/evidence-data.json", "/api/model"]:
                assert call("GET", path, cookie=operator)[0] == 403
                assert call("GET", path)[0] in {302, 401}
            assert call("GET", "/api/bearing/info", cookie=operator)[0] == 403
            assert call("POST", "/api/score", {"records": records()}, operator)[0] == 403
            for path in ["/models.html", "/shp.html", "/api/model"]:
                assert call("GET", path, cookie=engineer)[0] == 200
            assert call("POST", "/api/score", {"records": records()}, engineer)[0] == 200
            assert call("GET", "/api/runs", cookie=engineer)[0] == 200
        finally:
            connection.close()
            server.shutdown()
            worker.join(timeout=5)
