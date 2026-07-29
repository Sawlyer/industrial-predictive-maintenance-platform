"""API contract tests. The service must stay up even with no model on disk."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_always_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_a_loaded_model_or_returns_a_prediction(client):
    payload = {
        "unit": 1,
        "history": [
            {"cycle": c, "op_settings": [0.0, 0.0, 0.0], "sensors": [float(c)] * 21}
            for c in range(1, 26)
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["predicted_rul"] >= 0
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert body["risk_band"] in {"healthy", "watch", "warning", "critical"}


def test_predict_rejects_a_malformed_reading(client):
    payload = {"unit": 1, "history": [{"cycle": 1, "op_settings": [0.0], "sensors": [1.0]}]}
    assert client.post("/predict", json=payload).status_code == 422
