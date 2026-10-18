# BankNextAction AI — inference image (FastAPI + trained artifacts). No training happens here.
# Build:  docker build -t banknextaction .
# Run:    docker run --rm -p 8000:8000 banknextaction   ->  http://localhost:8000/docs
FROM python:3.11-slim

# uv pinned to an exact version, copied from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# UV_NO_CACHE: no package cache inside the image (the equivalent of pip --no-cache-dir).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencies from the lockfile only (cached layer; CPU-only torch on Linux).
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

# 2) Code, configuration and the trained artifacts the API serves (verified by SHA-256 at start).
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    BNA_CONFIG_PATH=/app/configs/config.yaml

# Run as an unprivileged user.
RUN useradd --system --no-create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "banknextaction.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
