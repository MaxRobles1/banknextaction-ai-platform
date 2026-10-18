"""Model B: a PyTorch MLP trained with an explicit loop (PDF §5.1.6).

The network receives the output of the same ColumnTransformer used by the baseline, fitted
on the training split only and saved next to the weights, so inference applies exactly the
training transformation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from banknextaction.config import Config
from banknextaction.features.preprocess import build_preprocessor
from banknextaction.models.base import BaseModel
from banknextaction.models.registry import register_model
from banknextaction.training.loop import (
    History,
    TrainingOutcome,
    fit_with_early_stopping,
    save_checkpoint,
)
from banknextaction.utils.io import atomic_write
from banknextaction.utils.logging import get_logger
from banknextaction.utils.timing import timed

log = get_logger(__name__)

PREPROCESSOR_FILE = "torch_preprocessor.joblib"
STATE_FILE = "mlp_state.pt"
CHECKPOINT_FILE = "mlp_checkpoint.pt"


class TabularDataset(Dataset):
    """Pre-transformed feature matrix and binary target as float32 tensors."""

    def __init__(self, features: np.ndarray, target: np.ndarray | None = None) -> None:
        # torch.tensor copies, so the dataset never aliases read-only pandas/NumPy buffers.
        self.features = torch.tensor(features, dtype=torch.float32)
        n_rows = len(self.features)
        values = np.zeros(n_rows) if target is None else np.asarray(target)
        self.target = torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.target[index]


class PropensityMLP(nn.Module):
    """Linear -> ReLU -> Dropout blocks, then one output logit per client."""

    def __init__(self, n_inputs: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch,); the sigmoid is applied outside (stable loss)."""
        return self.network(features).squeeze(-1)


@register_model("mlp")
class MlpModel(BaseModel):
    """BaseModel adapter around the preprocessor, the network and its training loop."""

    def __init__(
        self,
        cfg: Config | None = None,
        preprocessor: ColumnTransformer | None = None,
        network: PropensityMLP | None = None,
        architecture: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.network = network
        self.architecture = architecture
        self.history: History | None = None
        self.outcome: TrainingOutcome | None = None
        self._checkpoint_path: Path | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> MlpModel:
        return cls(cfg=cfg)

    def _loader(self, matrix: np.ndarray, target: pd.Series, shuffle: bool) -> DataLoader:
        assert self.cfg is not None
        generator = torch.Generator().manual_seed(self.cfg.project.seed)
        return DataLoader(
            TabularDataset(matrix, target.to_numpy()),
            batch_size=self.cfg.mlp.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    @timed
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> MlpModel:
        if self.cfg is None or x_val is None or y_val is None:
            raise RuntimeError("MlpModel needs a Config and a validation split to train")
        cfg = self.cfg
        torch.manual_seed(cfg.project.seed)

        self.preprocessor = build_preprocessor(cfg.features).fit(x_train)
        train_matrix = self.preprocessor.transform(x_train)
        val_matrix = self.preprocessor.transform(x_val)

        self.architecture = {
            "n_inputs": int(train_matrix.shape[1]),
            "hidden_dims": list(cfg.mlp.hidden_dims),
            "dropout": cfg.mlp.dropout,
        }
        self.network = PropensityMLP(
            self.architecture["n_inputs"], tuple(cfg.mlp.hidden_dims), cfg.mlp.dropout
        )
        # Positive class weight = negatives / positives in TRAIN (5.8 % positives).
        positives = float(y_train.sum())
        pos_weight = torch.tensor((len(y_train) - positives) / positives, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.mlp.learning_rate, weight_decay=cfg.mlp.weight_decay
        )
        checkpoint_path = cfg.path(cfg.artifacts.checkpoints_dir) / "mlp_best.pt"

        def write_best_checkpoint(epoch: int, metric: float, history: History) -> None:
            save_checkpoint(
                checkpoint_path,
                self.network,
                optimizer,
                epoch=epoch,
                best_metric=metric,
                extra={
                    "monitored_metric": "val_pr_auc",
                    "architecture": self.architecture,
                    "pos_weight": float(pos_weight),
                    "history": history.to_dict(),
                },
            )

        self.history, _, self.outcome = fit_with_early_stopping(
            self.network,
            self._loader(train_matrix, y_train, shuffle=True),
            self._loader(val_matrix, y_val, shuffle=False),
            loss_fn=loss_fn,
            optimizer=optimizer,
            max_epochs=cfg.mlp.max_epochs,
            patience=cfg.mlp.patience,
            on_improve=write_best_checkpoint,
        )
        self._checkpoint_path = checkpoint_path
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.network is None:
            raise RuntimeError("model is not fitted")
        matrix = torch.as_tensor(self.preprocessor.transform(features), dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            return torch.sigmoid(self.network(matrix)).numpy().astype(np.float64)

    def save(self, models_dir: Path) -> list[Path]:
        if self.preprocessor is None or self.network is None or self.architecture is None:
            raise RuntimeError("nothing to save: model is not fitted")
        preprocessor_path = models_dir / PREPROCESSOR_FILE
        with atomic_write(preprocessor_path, mode="wb") as handle:
            joblib.dump(self.preprocessor, handle)
        state_path = models_dir / STATE_FILE
        with atomic_write(state_path, mode="wb") as handle:
            torch.save(
                {"model_state": self.network.state_dict(), "architecture": self.architecture},
                handle,
            )
        written = [preprocessor_path, state_path]
        if self._checkpoint_path is not None and self._checkpoint_path.exists():
            checkpoint_path = models_dir / CHECKPOINT_FILE
            shutil.copyfile(self._checkpoint_path, checkpoint_path)
            written.append(checkpoint_path)
        return written

    @classmethod
    def load(cls, models_dir: Path) -> MlpModel:
        # Only artifacts produced by this project are loaded; torch uses weights_only=True.
        preprocessor = joblib.load(models_dir / PREPROCESSOR_FILE)
        state = torch.load(models_dir / STATE_FILE, map_location="cpu", weights_only=True)
        arch = state["architecture"]
        network = PropensityMLP(arch["n_inputs"], tuple(arch["hidden_dims"]), arch["dropout"])
        network.load_state_dict(state["model_state"])
        network.eval()
        return cls(preprocessor=preprocessor, network=network, architecture=arch)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        if self.architecture is not None:
            info["architecture"] = self.architecture
        return info

    def training_summary(self) -> dict[str, Any]:
        """Best/stop epochs and the full per-epoch history."""
        assert self.history is not None and self.outcome is not None
        return {
            "best_epoch": self.outcome.best_epoch,
            "stopped_epoch": self.outcome.stopped_epoch,
            "best_val_pr_auc": round(self.outcome.best_val_pr_auc, 4),
            "restored_best_weights": self.outcome.restored_best_weights,
            "history": self.history.to_dict(),
        }
