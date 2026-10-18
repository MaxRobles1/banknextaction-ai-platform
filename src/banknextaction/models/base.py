"""The contract every model implements (PDF §5.1.3).

Training, evaluation and the API only talk to BaseModel. That is what lets the same
evaluation code compare the scikit-learn baseline and the PyTorch network, and lets the API
serve either one without knowing which it is (dependency inversion, Clase 3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from banknextaction.config import Config


class BaseModel(ABC):
    """Binary propensity model: fit on pre-contact features, return P(subscribe)."""

    #: Registry key, set by @register_model.
    name: str = "base"

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Config) -> BaseModel:
        """Build an untrained model from the configuration (used by the factory)."""

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> BaseModel:
        """Train on the training split; the validation split is only used to monitor."""

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class, shape (n_rows,), values in [0, 1]."""

    @abstractmethod
    def save(self, models_dir: Path) -> list[Path]:
        """Persist everything needed for inference; return the written files."""

    @classmethod
    @abstractmethod
    def load(cls, models_dir: Path) -> BaseModel:
        """Rebuild a ready-to-predict model from saved artifacts, without retraining."""

    def predict(self, features: pd.DataFrame, threshold: float) -> np.ndarray:
        """Hard decision with a threshold chosen on validation."""
        return (self.predict_proba(features) >= threshold).astype(np.int8)

    def describe(self) -> dict[str, Any]:
        """Short metadata for reports and the API /health endpoint."""
        return {"name": self.name}
