"""FastAPI service: web UI at /, GET /health, POST /predict, GET /docs.

The model is loaded once, in the lifespan, before the first request, and its artifacts are
verified against the receipt. Requests are validated by Pydantic before reaching the model.
Run locally with:
    uv run uvicorn banknextaction.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from banknextaction import __version__
from banknextaction.api.schemas import ClientFeatures, HealthResponse, PredictionResponse
from banknextaction.api.service import ModelService
from banknextaction.config import load_config
from banknextaction.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

# Static scoring page (HTML/CSS/JS) shipped inside the package; it only calls /predict.
WEB_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.service = ModelService.load(load_config())
    yield
    app.state.service = None


app = FastAPI(
    title="BankNextAction AI",
    version=__version__,
    summary="Term-deposit propensity scoring from pre-contact client information.",
    description=(
        "Illustrative model for learning purposes (ITAM final project): not financial advice "
        "and not causal evidence of campaign effectiveness."
    ),
    lifespan=lifespan,
)


def _service(request: Request) -> ModelService:
    return request.app.state.service


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Scoring web page; it sends the form to the real POST /predict endpoint."""
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    """Service status, served model, version, artifact dates and hashes."""
    return _service(request).health()


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures, request: Request) -> PredictionResponse:
    """Probability that the client subscribes, and the decision at the validation threshold."""
    service = _service(request)
    result = service.predict(client.model_dump())
    log.info(
        "prediction served: class=%d latency_ms=%.2f",
        result.predicted_class,
        result.inference_time_ms,
    )
    return PredictionResponse(
        subscription_probability=round(result.probability, 6),
        non_subscription_probability=round(1.0 - result.probability, 6),
        predicted_class=result.predicted_class,
        decision_threshold=result.threshold,
        model_name=service.name,
        model_version=service.version,
        inference_time_ms=result.inference_time_ms,
    )
