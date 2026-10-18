"""PyTorch model: output shape (PDF §5.1.2), early stopping restore and resumable checkpoint."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from banknextaction.config import Config
from banknextaction.data.load import load_dataset
from banknextaction.models import create_model, model_class
from banknextaction.models.torch_model import CHECKPOINT_FILE, PropensityMLP, TabularDataset
from banknextaction.training.loop import EarlyStopping, load_checkpoint, train_one_epoch
from tests.conftest import make_raw_frame


@pytest.fixture
def small_cfg(cfg: Config, tmp_path: Path) -> Config:
    mlp = dataclasses.replace(cfg.mlp, max_epochs=4, patience=2, batch_size=64)
    artifacts = dataclasses.replace(cfg.artifacts, checkpoints_dir=tmp_path / "ckpt")
    return dataclasses.replace(cfg, mlp=mlp, artifacts=artifacts)


@pytest.fixture
def xy(tmp_path: Path, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    path = tmp_path / "raw.csv"
    make_raw_frame(n_rows=400, seed=2).to_csv(path, sep=";", index=False)
    return load_dataset(path, cfg.data, cfg.features)


def test_mlp_forward_returns_one_logit_per_row() -> None:
    net = PropensityMLP(n_inputs=7, hidden_dims=(8, 4), dropout=0.1)

    out = net(torch.zeros(5, 7))

    assert out.shape == (5,)


def test_dataset_returns_float32_pairs() -> None:
    data = TabularDataset(np.ones((3, 2)), np.array([0, 1, 0]))

    features, target = data[1]

    assert len(data) == 3
    assert features.dtype == torch.float32
    assert target.item() == 1.0


def test_early_stopping_restores_best_weights() -> None:
    net = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=2)
    stopper.step(1, 0.5, net)
    best = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        net.weight.add_(1.0)
    assert not stopper.step(2, 0.4, net)
    assert stopper.step(3, 0.3, net)

    stopper.restore(net)

    assert stopper.best_epoch == 1
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, best[key])


def test_mlp_predicts_probabilities_saves_and_reloads(
    xy: tuple[pd.DataFrame, pd.Series], small_cfg: Config, tmp_path: Path
) -> None:
    features, target = xy
    model = create_model("mlp", small_cfg)
    model.fit(features.iloc[:300], target.iloc[:300], features.iloc[300:], target.iloc[300:])

    proba = model.predict_proba(features.iloc[300:])
    written = {p.name for p in model.save(tmp_path / "models")}
    reloaded = model_class("mlp").load(tmp_path / "models")

    assert proba.shape == (100,)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(reloaded.predict_proba(features.iloc[300:]), proba, rtol=1e-6)
    assert written == {"torch_preprocessor.joblib", "mlp_state.pt", CHECKPOINT_FILE}
    assert model.outcome.restored_best_weights


def test_checkpoint_is_complete_and_resumable(
    xy: tuple[pd.DataFrame, pd.Series], small_cfg: Config, tmp_path: Path
) -> None:
    features, target = xy
    model = create_model("mlp", small_cfg)
    model.fit(features.iloc[:300], target.iloc[:300], features.iloc[300:], target.iloc[300:])
    model.save(tmp_path / "models")

    ckpt = load_checkpoint(tmp_path / "models" / CHECKPOINT_FILE)

    assert {"model_state", "optimizer_state", "epoch", "best_metric"} <= set(ckpt)
    assert ckpt["epoch"] == model.outcome.best_epoch
    arch = ckpt["architecture"]
    net = PropensityMLP(arch["n_inputs"], tuple(arch["hidden_dims"]), arch["dropout"])
    net.load_state_dict(ckpt["model_state"])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state"])
    matrix = model.preprocessor.transform(features.iloc[:300])
    loader = torch.utils.data.DataLoader(TabularDataset(matrix, target.iloc[:300]), batch_size=64)

    loss = train_one_epoch(net, loader, nn.BCEWithLogitsLoss(), optimizer)

    assert np.isfinite(loss)
