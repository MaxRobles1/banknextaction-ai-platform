"""Shared fixtures. Tests never need the real dataset: they build a small synthetic file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from banknextaction.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

_CATEGORIES = {
    "job": ["admin.", "blue-collar", "management", "unknown"],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["may", "jun", "jul"],
    "poutcome": ["unknown", "failure", "other", "success"],
}


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "config.yaml")


def make_raw_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the exact raw schema of bank-full.csv."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n_rows),
            "balance": rng.integers(-2000, 20000, n_rows),
            "day": rng.integers(1, 29, n_rows),
            "duration": rng.integers(0, 1000, n_rows),
            "campaign": rng.integers(1, 10, n_rows),
            "pdays": np.where(rng.random(n_rows) < 0.8, -1, rng.integers(1, 400, n_rows)),
            "previous": rng.integers(0, 5, n_rows),
        }
    )
    for column, values in _CATEGORIES.items():
        frame[column] = rng.choice(values, n_rows)
    frame["y"] = np.where(rng.random(n_rows) < 0.15, "yes", "no")
    return frame


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "bank-full.csv"
    make_raw_frame().to_csv(path, sep=";", index=False)
    return path
