"""Data manifest: the verifiable link between the raw file, the configuration and the splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from banknextaction.config import Config
from banknextaction.data.fetch import FetchResult
from banknextaction.data.split import SplitBoundaries, split_summary

CLEANING_CRITERIA: tuple[str, ...] = (
    "Raw file is immutable: verified by SHA-256 and never written back.",
    "Explicit dtypes: int64 for numeric columns, category for text columns.",
    "No null values in the raw file (checked at load time; loading fails otherwise).",
    "'unknown' is kept as its own category: it carries information (e.g. no previous campaign).",
    "pdays == -1 means 'never contacted before'; handled in the feature step with an indicator.",
    "Target y mapped to 1 (yes) / 0 (no).",
)


# Code that determines the data step output. Its fingerprint replaces a git commit id:
# a manifest committed together with this code cannot contain its own commit id, while a
# content hash is stable, recomputable from any checkout and changes only with this code.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_CODE_FILES: tuple[str, ...] = (
    "config.py",
    "data/fetch.py",
    "data/load.py",
    "data/split.py",
    "data/manifest.py",
)


def data_code_fingerprint(package_dir: Path = _PACKAGE_DIR) -> str:
    """SHA-256 over the data-step source files (path + content, in a fixed order)."""
    hasher = hashlib.sha256()
    for relative in DATA_CODE_FILES:
        hasher.update(relative.encode())
        hasher.update((package_dir / relative).read_bytes())
    return hasher.hexdigest()


def data_config_fingerprint(raw_config: dict[str, Any]) -> str:
    """SHA-256 of the data, features and split sections only (canonical JSON).

    Hashing these sections instead of the whole YAML keeps the manifest valid when
    unrelated settings (model hyperparameters) change later.
    """
    relevant = {key: raw_config[key] for key in ("data", "features", "split")}
    canonical = json.dumps(relevant, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_manifest(
    cfg: Config,
    fetch: FetchResult,
    raw: pd.DataFrame,
    target: pd.Series,
    bounds: SplitBoundaries,
) -> dict[str, Any]:
    """Assemble the manifest dictionary (no values are invented: all are measured)."""
    with cfg.source_file.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return {
        "dataset": {
            "name": cfg.data.source_name,
            "page": cfg.data.dataset_page,
            "download_url": cfg.data.download_url,
            "archive_members": list(cfg.data.archive_members),
            "license": cfg.data.license,
            "doi": cfg.data.doi,
            "citation": cfg.data.citation,
            "ordering": cfg.data.ordering,
            "retrieved_at": fetch.retrieved_at,
            "sha256": fetch.sha256,
            "rows": len(raw),
            "columns": raw.shape[1],
            "schema": {col: str(dtype) for col, dtype in raw.dtypes.items()},
        },
        "features": {
            "numeric": list(cfg.features.numeric),
            "categorical": list(cfg.features.categorical),
            "excluded": dict(cfg.features.excluded),
            "target": cfg.data.target,
            "positive_label": cfg.data.positive_label,
        },
        "cleaning": list(CLEANING_CRITERIA),
        "split": {
            "method": "chronological cut on row order (no shuffling)",
            "fractions": {
                "train": cfg.split.train_frac,
                "val": cfg.split.val_frac,
                "test": cfg.split.test_frac,
            },
            "boundaries": {k: list(v) for k, v in bounds.as_dict().items()},
            "summary": split_summary(target, bounds),
            "cv": f"TimeSeriesSplit(n_splits={cfg.split.cv_splits}) inside train only",
        },
        "reproducibility": {
            "seed": cfg.project.seed,
            "config_file": cfg.source_file.relative_to(cfg.root).as_posix(),
            "data_config_sha256": data_config_fingerprint(raw_config),
            "data_code_files": [f"src/banknextaction/{f}" for f in DATA_CODE_FILES],
            "data_code_sha256": data_code_fingerprint(),
            "regenerate_with": "uv run python scripts/prepare_data.py",
        },
    }
