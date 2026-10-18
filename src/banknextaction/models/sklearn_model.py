"""Model A: scikit-learn Pipeline (ColumnTransformer + LogisticRegression).

Hyperparameters are searched with GridSearchCV over a TimeSeriesSplit of the TRAINING rows
only: every fold trains on earlier rows and validates on later ones, and the whole pipeline
(preprocessing included) is refitted inside each fold, so no fold sees statistics from the
rows it is scored on (PDF §5.1.5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from banknextaction.config import Config
from banknextaction.features.preprocess import build_preprocessor
from banknextaction.models.base import BaseModel
from banknextaction.models.registry import register_model
from banknextaction.utils.io import atomic_write
from banknextaction.utils.logging import get_logger
from banknextaction.utils.timing import timed

log = get_logger(__name__)

ARTIFACT_NAME = "baseline_pipeline.joblib"
CV_SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
REFIT_METRIC = "pr_auc"


@dataclass(frozen=True)
class SearchResult:
    """What the grid search tried and what won (reported in the README)."""

    cv: str
    refit_metric: str
    best_params: dict[str, Any]
    best_cv_pr_auc: float
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@register_model("logreg")
class LogisticBaseline(BaseModel):
    """Baseline: one pipeline object holds preprocessing and the classifier."""

    def __init__(
        self,
        cfg: Config | None = None,
        pipeline: Pipeline | None = None,
        search: SearchResult | None = None,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.search = search

    @classmethod
    def from_config(cls, cfg: Config) -> LogisticBaseline:
        return cls(cfg=cfg)

    def _make_pipeline(self) -> Pipeline:
        assert self.cfg is not None
        classifier = LogisticRegression(
            max_iter=self.cfg.baseline.max_iter, random_state=self.cfg.project.seed
        )
        return Pipeline(
            [("preprocess", build_preprocessor(self.cfg.features)), ("model", classifier)]
        )

    @timed
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LogisticBaseline:
        if self.cfg is None:
            raise RuntimeError("LogisticBaseline needs a Config to be trained")
        folds = TimeSeriesSplit(n_splits=self.cfg.split.cv_splits)
        grid = GridSearchCV(
            self._make_pipeline(),
            param_grid=self.cfg.baseline.param_grid,
            scoring=CV_SCORING,
            refit=REFIT_METRIC,
            cv=folds,
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)
        self.pipeline = grid.best_estimator_
        self.search = _summarize(grid, folds)
        log.info(
            "best params %s, CV PR-AUC %.4f", self.search.best_params, self.search.best_cv_pr_auc
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return self.pipeline.predict_proba(features)[:, 1]

    def save(self, models_dir: Path) -> list[Path]:
        if self.pipeline is None:
            raise RuntimeError("nothing to save: model is not fitted")
        path = models_dir / ARTIFACT_NAME
        with atomic_write(path, mode="wb") as handle:
            joblib.dump(self.pipeline, handle)
        return [path]

    @classmethod
    def load(cls, models_dir: Path) -> LogisticBaseline:
        # joblib/pickle: only artifacts produced by this project are loaded, never user files.
        return cls(pipeline=joblib.load(models_dir / ARTIFACT_NAME))

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        if self.search is not None:
            info["best_params"] = self.search.best_params
        return info


def _summarize(grid: GridSearchCV, folds: TimeSeriesSplit) -> SearchResult:
    results = grid.cv_results_
    order = np.argsort(results["rank_test_pr_auc"])
    candidates = [
        {
            "params": {k: _plain(v) for k, v in results["params"][i].items()},
            "cv_pr_auc_mean": round(float(results["mean_test_pr_auc"][i]), 4),
            "cv_pr_auc_std": round(float(results["std_test_pr_auc"][i]), 4),
            "cv_roc_auc_mean": round(float(results["mean_test_roc_auc"][i]), 4),
            "rank": int(results["rank_test_pr_auc"][i]),
        }
        for i in order
    ]
    return SearchResult(
        cv=f"TimeSeriesSplit(n_splits={folds.n_splits}) on the training split",
        refit_metric="average_precision (PR-AUC)",
        best_params={k: _plain(v) for k, v in grid.best_params_.items()},
        best_cv_pr_auc=round(float(grid.best_score_), 4),
        candidates=candidates,
    )


def _plain(value: Any) -> Any:
    """Make numpy scalars JSON-friendly."""
    return value.item() if isinstance(value, np.generic) else value
