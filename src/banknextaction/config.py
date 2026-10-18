"""Typed configuration loaded from configs/config.yaml.

The YAML file is the single source of paths, features and hyperparameters. Loading it
builds frozen dataclasses and fails at startup if something does not add up, instead of
failing hours later in the middle of training.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


class ConfigError(ValueError):
    """The configuration file is missing a key or has an inconsistent value."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    model_version: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    source_name: str
    dataset_page: str
    download_url: str
    archive_members: tuple[str, ...]
    license: str
    doi: str
    citation: str
    ordering: str
    raw_sha256: str
    raw_file: Path
    manifest_file: Path
    separator: str
    target: str
    positive_label: str


@dataclass(frozen=True)
class FeatureConfig:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    excluded: dict[str, str]

    def __post_init__(self) -> None:
        overlap = set(self.numeric) & set(self.categorical)
        leaked = set(self.excluded) & (set(self.numeric) | set(self.categorical))
        if overlap:
            raise ConfigError(f"features both numeric and categorical: {sorted(overlap)}")
        if leaked:
            raise ConfigError(f"excluded features are also used as inputs: {sorted(leaked)}")

    @property
    def inputs(self) -> tuple[str, ...]:
        """Model input columns, numeric first, in a fixed order."""
        return self.numeric + self.categorical


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float
    val_frac: float
    test_frac: float
    cv_splits: int

    def __post_init__(self) -> None:
        fracs = (self.train_frac, self.val_frac, self.test_frac)
        if any(f <= 0 for f in fracs) or not math.isclose(sum(fracs), 1.0, abs_tol=1e-9):
            raise ConfigError(f"split fractions must be positive and sum to 1, got {fracs}")
        if self.cv_splits < 2:
            raise ConfigError("cv_splits must be at least 2")


@dataclass(frozen=True)
class BaselineConfig:
    param_grid: dict[str, list[Any]]
    max_iter: int


@dataclass(frozen=True)
class MlpConfig:
    hidden_dims: tuple[int, ...]
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class ThresholdConfig:
    criterion: str
    grid_size: int
    tie_break: str


@dataclass(frozen=True)
class ArtifactConfig:
    models_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    figures_dir: Path
    receipt_file: Path


@dataclass(frozen=True)
class Config:
    root: Path
    project: ProjectConfig
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    baseline: BaselineConfig
    mlp: MlpConfig
    threshold: ThresholdConfig
    artifacts: ArtifactConfig
    source_file: Path = field(compare=False)

    def path(self, relative: Path) -> Path:
        """Resolve a repository-relative path against the repository root."""
        return relative if relative.is_absolute() else self.root / relative


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        section = raw[name]
    except KeyError as exc:
        raise ConfigError(f"missing section '{name}' in the configuration") from exc
    if not isinstance(section, dict):
        raise ConfigError(f"section '{name}' must be a mapping")
    return section


def _build(
    cls: type, values: dict[str, Any], paths: tuple[str, ...] = (), seqs: tuple[str, ...] = ()
) -> Any:
    kwargs = dict(values)
    for key in paths:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    for key in seqs:
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"invalid keys for {cls.__name__}: {exc}") from exc


def load_config(path: Path | None = None) -> Config:
    """Load and validate the YAML configuration.

    The path comes from the argument, else BNA_CONFIG_PATH, else configs/config.yaml.
    Relative paths inside the file are resolved against the repository root, which is
    the parent of the configs/ folder.
    """
    config_path = Path(path or os.environ.get("BNA_CONFIG_PATH", DEFAULT_CONFIG_PATH)).resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("the configuration file must contain a mapping")

    return Config(
        root=config_path.parent.parent,
        project=_build(ProjectConfig, _section(raw, "project")),
        data=_build(
            DataConfig,
            _section(raw, "data"),
            paths=("raw_file", "manifest_file"),
            seqs=("archive_members",),
        ),
        features=_build(FeatureConfig, _section(raw, "features"), seqs=("numeric", "categorical")),
        split=_build(SplitConfig, _section(raw, "split")),
        baseline=_build(BaselineConfig, _section(raw, "baseline")),
        mlp=_build(MlpConfig, _section(raw, "mlp"), seqs=("hidden_dims",)),
        threshold=_build(ThresholdConfig, _section(raw, "threshold")),
        artifacts=_build(
            ArtifactConfig,
            _section(raw, "artifacts"),
            paths=("models_dir", "checkpoints_dir", "reports_dir", "figures_dir", "receipt_file"),
        ),
        source_file=config_path,
    )
