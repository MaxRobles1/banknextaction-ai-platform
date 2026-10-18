"""Load the raw CSV with explicit dtypes and validate its schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from banknextaction.config import DataConfig, FeatureConfig
from banknextaction.utils.logging import get_logger

log = get_logger(__name__)

RAW_NUMERIC: tuple[str, ...] = (
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
)
RAW_CATEGORICAL: tuple[str, ...] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
)
RAW_COLUMNS: frozenset[str] = frozenset((*RAW_NUMERIC, *RAW_CATEGORICAL, "y"))


class SchemaError(ValueError):
    """The raw file does not have the expected columns or values."""


def load_raw(path: Path, separator: str = ";") -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes: int64 for numbers, category for text.

    'unknown' is kept as a regular category on purpose: in this dataset it is information
    (for example, poutcome='unknown' means there was no previous campaign), not a gap.
    """
    dtypes: dict[str, str] = {c: "int64" for c in RAW_NUMERIC}
    dtypes.update({c: "category" for c in RAW_CATEGORICAL})
    dtypes["y"] = "category"
    frame = pd.read_csv(path, sep=separator, dtype=dtypes)

    missing = RAW_COLUMNS - set(frame.columns)
    extra = set(frame.columns) - RAW_COLUMNS
    if missing or extra:
        raise SchemaError(f"unexpected schema: missing={sorted(missing)} extra={sorted(extra)}")
    n_null = int(frame.isna().sum().sum())
    if n_null:
        raise SchemaError(f"raw data has {n_null} null values; expected none")
    log.info("loaded %s rows x %s columns from %s", len(frame), frame.shape[1], path.name)
    return frame


def make_target(frame: pd.DataFrame, data_cfg: DataConfig) -> pd.Series:
    """Binary target as int8: 1 when the client subscribed (y == positive_label)."""
    labels = set(frame[data_cfg.target].cat.categories)
    if data_cfg.positive_label not in labels or len(labels) != 2:
        raise SchemaError(f"target must be binary with '{data_cfg.positive_label}', got {labels}")
    return (frame[data_cfg.target] == data_cfg.positive_label).astype("int8").rename("target")


def select_inputs(frame: pd.DataFrame, features: FeatureConfig) -> pd.DataFrame:
    """Keep only the pre-contact input columns, as an independent copy (no view chaining)."""
    return frame.loc[:, list(features.inputs)].copy()


def load_dataset(
    path: Path, data_cfg: DataConfig, features: FeatureConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Raw CSV -> (inputs X, binary target y), with row order preserved."""
    frame = load_raw(path, data_cfg.separator)
    return select_inputs(frame, features), make_target(frame, data_cfg)
