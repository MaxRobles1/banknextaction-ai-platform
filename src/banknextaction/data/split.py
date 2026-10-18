"""Chronological train / validation / test split.

bank-full.csv is ordered by date, so the split cuts the row order instead of shuffling:
the model is asked the same question it will face in production, predicting later
campaigns from earlier ones (PDF §5.1.5 and §5.1.6). Rows are independent clients and the
inputs are not overlapping windows, so the gap used for windowed series in Clase 9 is not
needed here: no input of one row contains the label of another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitBoundaries:
    """Half-open row intervals [start, end) for each split."""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return asdict(self)

    def sizes(self) -> dict[str, int]:
        return {name: end - start for name, (start, end) in self.as_dict().items()}


def chronological_boundaries(n_rows: int, train_frac: float, val_frac: float) -> SplitBoundaries:
    """Cut points for a chronological split; every part must be non-empty."""
    train_end = int(np.floor(n_rows * train_frac))
    val_end = int(np.floor(n_rows * (train_frac + val_frac)))
    bounds = SplitBoundaries(train=(0, train_end), val=(train_end, val_end), test=(val_end, n_rows))
    if min(bounds.sizes().values()) <= 0:
        raise ValueError(f"empty split for n_rows={n_rows}: {bounds.sizes()}")
    return bounds


def apply_split(
    features: pd.DataFrame, target: pd.Series, bounds: SplitBoundaries
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Slice X and y by position (iloc), keeping the original order inside each split."""
    if len(features) != len(target):
        raise ValueError(f"X has {len(features)} rows but y has {len(target)}")
    return {
        name: (features.iloc[start:end].copy(), target.iloc[start:end].copy())
        for name, (start, end) in bounds.as_dict().items()
    }


def split_summary(target: pd.Series, bounds: SplitBoundaries) -> dict[str, dict[str, float]]:
    """Size, prevalence and majority-class floor of each split (computed with NumPy)."""
    values = target.to_numpy()
    summary: dict[str, dict[str, float]] = {}
    for name, (start, end) in bounds.as_dict().items():
        prevalence = float(values[start:end].mean())
        summary[name] = {
            "rows": end - start,
            "positives": int(values[start:end].sum()),
            "prevalence": round(prevalence, 4),
            "majority_class_accuracy": round(max(prevalence, 1.0 - prevalence), 4),
        }
    return summary
