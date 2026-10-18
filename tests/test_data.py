"""Data loading, chronological split and integrity checks (PDF §5.1.2: 'la carga de datos')."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from banknextaction.config import Config, ConfigError, SplitConfig
from banknextaction.data.fetch import DataIntegrityError, verify_sha256
from banknextaction.data.load import SchemaError, load_dataset, load_raw
from banknextaction.data.manifest import data_code_fingerprint, data_config_fingerprint
from banknextaction.data.split import apply_split, chronological_boundaries, split_summary
from banknextaction.utils.hashing import sha256_file
from banknextaction.utils.io import atomic_write, read_json


def test_load_dataset_returns_precontact_inputs_and_binary_target(
    raw_csv: Path, cfg: Config
) -> None:
    features, target = load_dataset(raw_csv, cfg.data, cfg.features)

    assert list(features.columns) == list(cfg.features.inputs)
    assert not set(cfg.features.excluded) & set(features.columns)
    assert target.dtype == np.int8
    assert set(target.unique()) <= {0, 1}
    assert all(isinstance(features[c].dtype, pd.CategoricalDtype) for c in cfg.features.categorical)


def test_load_raw_rejects_missing_column(raw_csv: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.csv"
    pd.read_csv(raw_csv, sep=";").drop(columns=["poutcome"]).to_csv(broken, sep=";", index=False)

    with pytest.raises(SchemaError, match="poutcome"):
        load_raw(broken)


def test_chronological_split_keeps_order_without_overlap() -> None:
    bounds = chronological_boundaries(1000, train_frac=0.7, val_frac=0.15)
    features = pd.DataFrame({"row": np.arange(1000)})
    target = pd.Series(np.zeros(1000, dtype=np.int8))

    parts = apply_split(features, target, bounds)

    assert bounds.sizes() == {"train": 700, "val": 150, "test": 150}
    assert parts["train"][0]["row"].max() < parts["val"][0]["row"].min()
    assert parts["val"][0]["row"].max() < parts["test"][0]["row"].min()
    assert sum(len(x) for x, _ in parts.values()) == 1000


def test_split_summary_reports_majority_class_floor() -> None:
    target = pd.Series([0, 0, 0, 1] * 25, dtype=np.int8)
    bounds = chronological_boundaries(100, 0.6, 0.2)

    summary = split_summary(target, bounds)

    assert summary["train"]["prevalence"] == pytest.approx(0.25)
    assert summary["train"]["majority_class_accuracy"] == pytest.approx(0.75)


def test_split_config_rejects_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ConfigError):
        SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.2, cv_splits=5)


def test_verify_sha256_detects_modified_file(raw_csv: Path) -> None:
    expected = sha256_file(raw_csv)
    assert verify_sha256(raw_csv, expected) == expected

    raw_csv.write_text(raw_csv.read_text() + "tampered\n")
    with pytest.raises(DataIntegrityError):
        verify_sha256(raw_csv, expected)


def test_atomic_write_keeps_previous_file_when_writing_fails(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("original")

    with pytest.raises(RuntimeError), atomic_write(target) as handle:
        handle.write("partial")
        raise RuntimeError("boom")

    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_committed_manifest_matches_current_data_code_and_config(cfg: Config) -> None:
    """A stale manifest (data code or data config changed without regenerating) fails here."""
    manifest = read_json(cfg.path(cfg.data.manifest_file))
    with cfg.source_file.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    reproducibility = manifest["reproducibility"]
    assert reproducibility["data_code_sha256"] == data_code_fingerprint()
    assert reproducibility["data_config_sha256"] == data_config_fingerprint(raw_config)
    assert manifest["dataset"]["sha256"] == cfg.data.raw_sha256
