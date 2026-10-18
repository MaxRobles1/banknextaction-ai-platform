"""Feature engineering as a scikit-learn ColumnTransformer.

Everything is fitted inside the pipeline, so scaling statistics and categories are learned
from the training rows of each fold only, never from validation or test (PDF §5.1.5).
The same fitted transformer feeds the PyTorch model, which guarantees that both models and
the API use exactly the same transformation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from banknextaction.config import FeatureConfig

# pdays == -1 is a sentinel for "never contacted in a previous campaign", not a number.
PDAYS_COLUMN = "pdays"
PDAYS_NEVER = -1
# Heavy-tailed counts and amounts (balance reaches ~100k with a median of ~450).
LOG_SCALED_COLUMNS: frozenset[str] = frozenset({"balance", "previous"})


def signed_log1p(values: ArrayLike) -> np.ndarray:
    """sign(x) * log(1 + |x|): compresses long tails and keeps negative balances negative."""
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.log1p(np.abs(array))


def contact_history(values: ArrayLike) -> np.ndarray:
    """Turn pdays into [previously_contacted, log1p(days since contact, 0 if never)]."""
    pdays = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    contacted = (pdays != PDAYS_NEVER).astype(np.float64)
    days = np.where(contacted == 1.0, np.log1p(np.clip(pdays, 0, None)), 0.0)
    return np.hstack([contacted, days])


def _contact_history_names(_transformer: object, _input_features: object) -> np.ndarray:
    return np.array(["previously_contacted", "log_days_since_contact"], dtype=object)


def build_preprocessor(features: FeatureConfig) -> ColumnTransformer:
    """Numeric scaling (with log for long tails), pdays decoding and one-hot categories."""
    numeric = list(features.numeric)
    plain = [c for c in numeric if c != PDAYS_COLUMN and c not in LOG_SCALED_COLUMNS]
    logged = [c for c in numeric if c in LOG_SCALED_COLUMNS]

    transformers: list[tuple[str, object, list[str]]] = []
    if plain:
        transformers.append(("num", StandardScaler(), plain))
    if logged:
        log_scale = Pipeline(
            [
                ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num_log", log_scale, logged))
    if PDAYS_COLUMN in numeric:
        history = Pipeline(
            [
                (
                    "decode",
                    FunctionTransformer(contact_history, feature_names_out=_contact_history_names),
                ),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("contact", history, [PDAYS_COLUMN]))
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            list(features.categorical),
        )
    )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
