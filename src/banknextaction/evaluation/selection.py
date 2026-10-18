"""Choose the served model using VALIDATION results only (docs/DECISIONS.md D-005).

Order of criteria: F1 on validation, then PR-AUC on validation, then measured inference
latency, then artifact size. Each criterion only breaks ties of the previous one, where a
tie means a difference smaller than `tolerance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from banknextaction.models.base import BaseModel
from banknextaction.utils.timing import timed


@dataclass(frozen=True)
class Candidate:
    name: str
    val_f1: float
    val_pr_auc: float
    latency_ms_p50: float
    artifact_bytes: int


@dataclass(frozen=True)
class Selection:
    served: str
    reason: str
    ranking: list[dict[str, Any]]


def measure_latency_ms(
    model: BaseModel, sample: pd.DataFrame, repeats: int = 200, warmup: int = 20
) -> dict[str, float]:
    """Single-row predict_proba latency: median and p95 after warm-up (in-process, CPU)."""

    @timed
    def predict_one(row: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(row)

    rows = [sample.iloc[[i % len(sample)]] for i in range(repeats + warmup)]
    timings = []
    for i, row in enumerate(rows):
        predict_one(row)
        if i >= warmup:
            timings.append(predict_one.last_elapsed_ms)
    values = np.asarray(timings)
    return {
        "p50_ms": round(float(np.median(values)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "repeats": repeats,
        "warmup": warmup,
    }


def select_served_model(candidates: list[Candidate], tolerance: float = 1e-3) -> Selection:
    """Apply the criteria in order; never looks at test metrics."""

    def key(c: Candidate) -> tuple[float, float, float, int]:
        # Round to the tolerance so tiny differences count as ties for the next criterion.
        return (
            -round(c.val_f1 / tolerance),
            -round(c.val_pr_auc / tolerance),
            c.latency_ms_p50,
            c.artifact_bytes,
        )

    ranked = sorted(candidates, key=key)
    best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None
    if runner_up is None:
        reason = "only candidate"
    elif round(best.val_f1 / tolerance) != round(runner_up.val_f1 / tolerance):
        reason = f"higher validation F1 ({best.val_f1:.4f} vs {runner_up.val_f1:.4f})"
    elif round(best.val_pr_auc / tolerance) != round(runner_up.val_pr_auc / tolerance):
        reason = "F1 tie; higher validation PR-AUC"
    elif best.latency_ms_p50 != runner_up.latency_ms_p50:
        reason = "F1 and PR-AUC tie; lower latency"
    else:
        reason = "F1, PR-AUC and latency tie; smaller artifacts"
    return Selection(served=best.name, reason=reason, ranking=[vars(c) for c in ranked])
