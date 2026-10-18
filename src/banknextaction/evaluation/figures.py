"""Report figures saved under reports/figures (titles, axis labels and legends included)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from banknextaction.data.split import SplitBoundaries

_SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def plot_eda(raw: pd.DataFrame, target: pd.Series, bounds: SplitBoundaries, path: Path) -> Path:
    """Figure 1: subscription rate over time with the split cuts, and by previous outcome."""
    fig, (ax_time, ax_cat) = plt.subplots(1, 2, figsize=(13, 4.5))

    window = 1000
    rolling = target.rolling(window, min_periods=window // 2).mean().to_numpy()
    ax_time.plot(
        np.arange(len(rolling)),
        rolling,
        color="black",
        lw=1.2,
        label=f"rolling mean ({window} rows)",
    )
    for name, (start, end) in bounds.as_dict().items():
        share = float(target.iloc[start:end].mean())
        ax_time.axvspan(
            start, end, color=_SPLIT_COLORS[name], alpha=0.15, label=f"{name}: {share:.1%}"
        )
    ax_time.set_title("Subscription rate along time (rows ordered by date, 2008-2010)")
    ax_time.set_xlabel("Row position (chronological)")
    ax_time.set_ylabel("Share of clients with y = yes")
    ax_time.legend(loc="upper left", fontsize=8)

    rates = target.groupby(raw["poutcome"], observed=True).agg(["mean", "size"]).sort_values("mean")
    bars = ax_cat.barh(rates.index.astype(str), rates["mean"], color="#4C72B0")
    for bar, n in zip(bars, rates["size"], strict=True):
        ax_cat.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={n:,}",
            va="center",
            fontsize=8,
        )
    ax_cat.set_title("Subscription rate by outcome of the previous campaign")
    ax_cat.set_xlabel("Share of clients with y = yes")
    ax_cat.set_ylabel("poutcome")
    ax_cat.set_xlim(0, 1)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_learning_curves(training: dict[str, Any], path: Path) -> Path:
    """Figure 2: train/val loss per epoch and validation PR-AUC with the restored epoch."""
    history = training["history"]
    epochs = history["epoch"]
    best, stopped = training["best_epoch"], training["stopped_epoch"]
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_loss.plot(epochs, history["train_loss"], marker="o", ms=3, label="train loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", ms=3, label="validation loss")
    ax_loss.axvline(best, color="green", ls="--", label=f"best epoch {best} (restored)")
    ax_loss.set_title("MLP learning curves: weighted BCE loss per epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (BCE with positive-class weight)")
    ax_loss.legend()

    ax_metric.plot(
        epochs, history["val_pr_auc"], marker="o", ms=3, color="#C44E52", label="val PR-AUC"
    )
    ax_metric.plot(
        epochs, history["val_roc_auc"], marker="o", ms=3, color="#8172B2", label="val ROC-AUC"
    )
    ax_metric.axvline(best, color="green", ls="--", label=f"best epoch {best}")
    ax_metric.axvline(stopped, color="gray", ls=":", label=f"stopped at epoch {stopped}")
    ax_metric.set_title("Validation metrics per epoch (early stopping on PR-AUC)")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
