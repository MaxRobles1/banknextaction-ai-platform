"""Metrics, floors and threshold selection, shared by both models (same code, same numbers)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    f1: float
    criterion: str
    tie_break: str


@dataclass(frozen=True)
class MetricReport:
    """Metrics of one model on one split, next to the floors they must beat."""

    split: str
    rows: int
    prevalence: float
    threshold: float
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    floor_accuracy_majority: float
    floor_pr_auc_prevalence: float
    floor_roc_auc_random: float = 0.5

    def to_dict(self) -> dict[str, float | int | str]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def threshold_grid(size: int) -> np.ndarray:
    """Evenly spaced candidate thresholds strictly inside (0, 1)."""
    return np.linspace(0.0, 1.0, size + 2)[1:-1]


def f1_by_threshold(y_true: ArrayLike, proba: ArrayLike, thresholds: np.ndarray) -> np.ndarray:
    """F1 for every threshold at once (vectorized: thresholds x rows boolean matrix)."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(proba, dtype=np.float64)
    predicted = p[np.newaxis, :] >= thresholds[:, np.newaxis]
    tp = (predicted & y).sum(axis=1)
    fp = (predicted & ~y).sum(axis=1)
    fn = (~predicted & y).sum(axis=1)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros(len(thresholds)), where=denominator > 0)


def choose_threshold_f1(
    y_val: ArrayLike, proba_val: ArrayLike, grid_size: int, tie_break: str = "highest"
) -> ThresholdChoice:
    """Threshold that maximizes F1 on VALIDATION; ties resolved deterministically.

    tie_break="highest" keeps the largest threshold among equal-F1 candidates (fewer false
    positives for the same F1). The test split is never passed here.
    """
    if tie_break != "highest":
        raise ValueError(f"unsupported tie_break: {tie_break}")
    thresholds = threshold_grid(grid_size)
    scores = f1_by_threshold(y_val, proba_val, thresholds)
    best = np.flatnonzero(scores == scores.max())[-1]
    return ThresholdChoice(
        threshold=float(thresholds[best]),
        f1=float(scores[best]),
        criterion="max F1 on validation",
        tie_break=tie_break,
    )


def evaluate(split: str, y_true: ArrayLike, proba: ArrayLike, threshold: float) -> MetricReport:
    """All reported metrics for one split at a fixed threshold, with their floors."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(proba, dtype=np.float64)
    predicted = (p >= threshold).astype(np.int8)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(y.mean())
    return MetricReport(
        split=split,
        rows=len(y),
        prevalence=prevalence,
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=float((predicted == y).mean()),
        floor_accuracy_majority=max(prevalence, 1.0 - prevalence),
        floor_pr_auc_prevalence=prevalence,
    )
