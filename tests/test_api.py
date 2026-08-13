"""API contract tests. These run without a trained model artefact."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_always_answers():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_impossible_input():
    r = client.post("/predict", json={"location": "hebbal", "bedroom": 0, "bath": 2, "built_up_area": 1200})
    assert r.status_code == 422


def test_predict_rejects_absurd_area():
    r = client.post("/predict", json={"location": "hebbal", "bedroom": 3, "bath": 2, "built_up_area": 10})
    assert r.status_code == 422


def test_batch_rejects_empty():
    assert client.post("/predict/batch", json=[]).status_code == 422


@pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parents[1] / "models" / "price_model.joblib").exists(),
    reason="model artefact absent; run `make train`",
)
def test_predict_returns_a_plausible_price():
    r = client.post(
        "/predict",
        json={"location": "whitefield", "bedroom": 3, "bath": 2, "balcony": 1, "built_up_area": 1400},
    )
    assert r.status_code == 200
    price = r.json()["price_crore"]
    assert 0.05 < price < 30.0, f"implausible prediction: {price} crore"
