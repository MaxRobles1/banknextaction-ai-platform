"""Web UI (E9): served at /, static assets load, and it only talks to the real API."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from banknextaction.api.app import WEB_DIR, app
from banknextaction.api.schemas import ClientFeatures


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_the_scoring_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "BankNextAction AI" in response.text
    assert 'src="/static/app.js"' in response.text
    assert 'href="/static/styles.css"' in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [("/static/app.js", "javascript"), ("/static/styles.css", "text/css")],
)
def test_static_assets_are_served(client: TestClient, path: str, content_type: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_api_endpoints_keep_working_next_to_the_page(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_page_builds_the_form_from_the_real_contract(client: TestClient) -> None:
    """The UI reads ClientFeatures from /openapi.json and posts to /predict (no local model)."""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ClientFeatures"]

    assert 'SCHEMA_NAME = "ClientFeatures"' in script
    assert 'fetch("/predict"' in script
    assert set(schema["properties"]) == set(ClientFeatures.model_fields)


def test_synthetic_examples_in_the_page_are_valid_api_inputs(client: TestClient) -> None:
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const EXAMPLES = \[(.*?)\];", script, flags=re.S).group(1)
    examples = re.findall(r"\{(.*?)\}", block, flags=re.S)

    assert examples
    for body in examples:
        payload = dict(re.findall(r"(\w+): (\"[^\"]*\"|-?\d+)", body))
        parsed = {k: v.strip('"') if v.startswith('"') else int(v) for k, v in payload.items()}
        response = client.post("/predict", json=parsed)
        assert response.status_code == 200, parsed
