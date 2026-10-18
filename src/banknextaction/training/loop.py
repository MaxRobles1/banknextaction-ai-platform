"""PyTorch training loop: two halves per epoch, history, early stopping and checkpoints.

Each epoch has a half that learns (model.train(), gradients on) and a half that only looks
(model.eval() inside torch.no_grad()). Early stopping keeps a copy of the best weights and
RESTORES them at the end: stopping without restoring would keep the weights of the last,
already-worse epoch (PDF §5.1.6).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from banknextaction.utils.io import atomic_write
from banknextaction.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class History:
    """Per-epoch record used for the learning curves."""

    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)

    def append(self, epoch: int, train_loss: float, val_loss: float, pr: float, roc: float) -> None:
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_pr_auc.append(pr)
        self.val_roc_auc.append(roc)

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return asdict(self)


@dataclass
class EarlyStopping:
    """Stop after `patience` epochs without improving the monitored metric (higher is better)."""

    patience: int
    best_metric: float = -np.inf
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs: int = 0

    def step(self, epoch: int, metric: float, model: nn.Module) -> bool:
        """Record the epoch; return True when training should stop."""
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best weights back into the model."""
        if self.best_state is None:
            raise RuntimeError("no best state recorded")
        model.load_state_dict(self.best_state)


@dataclass(frozen=True)
class TrainingOutcome:
    best_epoch: int
    stopped_epoch: int
    best_val_pr_auc: float
    restored_best_weights: bool


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    """First half: learn. Returns the mean training loss of the epoch."""
    model.train()
    total, count = 0.0, 0
    for features, target in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
        count += len(target)
    return total / count


def evaluate_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module
) -> tuple[float, np.ndarray, np.ndarray]:
    """Second half: only look. Returns mean loss, probabilities and targets."""
    model.eval()
    total, count = 0.0, 0
    probas, targets = [], []
    with torch.no_grad():
        for features, target in loader:
            logits = model(features)
            total += loss_fn(logits, target).item() * len(target)
            count += len(target)
            probas.append(torch.sigmoid(logits).numpy())
            targets.append(target.numpy())
    return total / count, np.concatenate(probas), np.concatenate(targets)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_metric: float,
    extra: dict[str, Any],
) -> Path:
    """Complete checkpoint: weights, optimizer state, epoch and best metric (resumable)."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        **extra,
    }
    with atomic_write(path, mode="wb") as handle:
        torch.save(payload, handle)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a checkpoint written by save_checkpoint (tensors and plain Python only)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    patience: int,
    on_improve: Callable[[int, float, History], None] | None = None,
) -> tuple[History, EarlyStopping, TrainingOutcome]:
    """Run epochs until early stopping triggers, then restore the best weights.

    `on_improve(epoch, metric, history)` runs whenever validation improves; it is used to
    write the checkpoint at the best epoch, so weights and optimizer state stay consistent.
    """
    history = History()
    stopper = EarlyStopping(patience=patience)
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, proba, target = evaluate_epoch(model, val_loader, loss_fn)
        pr_auc = float(average_precision_score(target, proba))
        roc_auc = float(roc_auc_score(target, proba))
        history.append(epoch, train_loss, val_loss, pr_auc, roc_auc)
        log.info(
            "epoch %3d train_loss=%.4f val_loss=%.4f val_pr_auc=%.4f val_roc_auc=%.4f",
            epoch,
            train_loss,
            val_loss,
            pr_auc,
            roc_auc,
        )
        should_stop = stopper.step(epoch, pr_auc, model)
        if stopper.best_epoch == epoch and on_improve is not None:
            on_improve(epoch, pr_auc, history)
        if should_stop:
            break
    stopper.restore(model)
    outcome = TrainingOutcome(
        best_epoch=stopper.best_epoch,
        stopped_epoch=epoch,
        best_val_pr_auc=stopper.best_metric,
        restored_best_weights=True,
    )
    log.info(
        "stopped at epoch %d; restored weights of best epoch %d (val PR-AUC %.4f)",
        outcome.stopped_epoch,
        outcome.best_epoch,
        outcome.best_val_pr_auc,
    )
    return history, stopper, outcome
