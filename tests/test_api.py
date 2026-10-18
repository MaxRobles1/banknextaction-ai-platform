"""API contract (PDF §5.1.2: 'el contrato de la API'), using the committed artifacts."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from banknextaction.api.app import app
from banknextaction.api.schemas import EXAMPLE_CLIENT
from banknextaction.config import Config
from banknextaction.models.registry import model_class
from banknextaction.utils.hashing import md5_file
from banknextaction.utils.io import read_json

RESPONSE_KEYS = {
    "subscription_probability",
    "non_subscription_probability",
    "predicted_class",
    "decision_threshold",
    "model_name",
    "model_version",
    "inference_time_ms",
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:  # the `with` runs the lifespan: model loaded once
        yield test_client


def test_health_reports_served_model_and_matching_hashes(client: TestClient, cfg: Config) -> None:
    response = client.get("/health")

    body = response.json()
    receipt = read_json(cfg.path(cfg.artifacts.receipt_file))
    assert response.status_code == 200
    assert body["model_name"] == receipt["served_model"]
    assert body["decision_threshold"] == receipt["decision_threshold"]
    for artifact in body["artifacts"]:
        path = cfg.path(cfg.artifacts.models_dir) / artifact["file"]
        assert artifact["md5"] == md5_file(path)


def test_predict_returns_the_promised_contract(client: TestClient) -> None:
    response = client.post("/predict", json=EXAMPLE_CLIENT)

    body = response.json()
    assert response.status_code == 200
    assert set(body) == RESPONSE_KEYS
    assert 0.0 <= body["subscription_probability"] <= 1.0
    assert body["subscription_probability"] + body["non_subscription_probability"] == (
        pytest.approx(1.0)
    )
    assert body["predicted_class"] == int(
        body["subscription_probability"] >= body["decision_threshold"]
    )


def test_predict_matches_the_model_called_directly(client: TestClient, cfg: Config) -> None:
    served = client.get("/health").json()["model_name"]
    model = model_class(served).load(cfg.path(cfg.artifacts.models_dir))
    order = read_json(cfg.path(cfg.artifacts.receipt_file))["input_schema"]["order"]
    expected = float(model.predict_proba(pd.DataFrame([EXAMPLE_CLIENT], columns=order))[0])

    body = client.post("/predict", json=EXAMPLE_CLIENT).json()

    assert body["subscription_probability"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "change",
    [
        {"age": 300},
        {"pdays": -5},
        {"job": "astronaut"},
        {"income": 1000},
    ],
    ids=["age-out-of-range", "pdays-below-sentinel", "unknown-category", "extra-field"],
)
def test_predict_rejects_invalid_input_with_422(client: TestClient, change: dict) -> None:
    response = client.post("/predict", json={**EXAMPLE_CLIENT, **change})

    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    incomplete = {k: v for k, v in EXAMPLE_CLIENT.items() if k != "poutcome"}

    assert client.post("/predict", json=incomplete).status_code == 422


def test_schema_categories_match_the_training_categories(cfg: Config) -> None:
    from banknextaction.api.schemas import ClientFeatures  # noqa: PLC0415

    receipt = read_json(cfg.path(cfg.artifacts.receipt_file))
    for column, allowed in receipt["input_schema"]["categorical"].items():
        annotation = ClientFeatures.model_fields[column].annotation
        assert set(annotation.__args__) == set(allowed), column
