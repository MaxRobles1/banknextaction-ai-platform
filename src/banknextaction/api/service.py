"""Loads the served model once and answers predictions (no training, no refitting here)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import banknextaction.models  # noqa: F401  (registers the model classes)
from banknextaction.config import Config
from banknextaction.evaluation.receipt import verify_receipt
from banknextaction.models.base import BaseModel
from banknextaction.models.registry import model_class
from banknextaction.utils.io import read_json
from banknextaction.utils.logging import get_logger
from banknextaction.utils.timing import timed

log = get_logger(__name__)


class ArtifactIntegrityError(RuntimeError):
    """An artifact on disk does not match the SHA-256 recorded in the receipt."""


@dataclass(frozen=True)
class Prediction:
    probability: float
    predicted_class: int
    threshold: float
    inference_time_ms: float


class ModelService:
    """Verified, ready-to-serve model plus its receipt."""

    def __init__(self, cfg: Config, receipt: dict[str, Any], model: BaseModel) -> None:
        self.cfg = cfg
        self.receipt = receipt
        self.model = model
        self.name: str = receipt["served_model"]
        self.version: str = receipt["model_version"]
        self.threshold: float = float(receipt["decision_threshold"])
        self.input_order: list[str] = list(receipt["input_schema"]["order"])

    @classmethod
    def load(cls, cfg: Config) -> ModelService:
        """Verify every artifact hash against the receipt, then load the served model."""
        receipt_path = cfg.path(cfg.artifacts.receipt_file)
        mismatched = verify_receipt(receipt_path, cfg.root)
        if mismatched:
            raise ArtifactIntegrityError(f"artifacts do not match the receipt: {mismatched}")
        receipt = read_json(receipt_path)
        model = model_class(receipt["served_model"]).load(cfg.path(cfg.artifacts.models_dir))
        log.info(
            "model loaded once: %s v%s (threshold %.3f, artifacts verified by SHA-256)",
            receipt["served_model"],
            receipt["model_version"],
            receipt["decision_threshold"],
        )
        return cls(cfg, receipt, model)

    @timed
    def _predict_proba(self, frame: pd.DataFrame) -> float:
        return float(self.model.predict_proba(frame)[0])

    def predict(self, client: dict[str, Any]) -> Prediction:
        """One client -> probability, class at the validation threshold, measured latency."""
        frame = pd.DataFrame([client], columns=self.input_order)
        probability = self._predict_proba(frame)
        return Prediction(
            probability=probability,
            predicted_class=int(probability >= self.threshold),
            threshold=self.threshold,
            inference_time_ms=round(type(self)._predict_proba.last_elapsed_ms, 3),
        )

    def health(self) -> dict[str, Any]:
        served = self.receipt["models"][self.name]
        return {
            "status": "ok",
            "model_name": self.name,
            "model_version": self.version,
            "decision_threshold": self.threshold,
            "artifact_created_at": max(a["created_at"] for a in served["artifacts"]),
            "receipt_generated_at": self.receipt["generated_at"],
            "artifacts": [
                {"file": Path(a["file"]).name, "sha256": a["sha256"], "md5": a["md5"]}
                for a in served["artifacts"]
            ],
        }
