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
