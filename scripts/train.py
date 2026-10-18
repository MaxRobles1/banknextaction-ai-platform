"""Train the registered models on TRAIN, choose each threshold on VALIDATION, save artifacts.

The test split is not loaded into any model here: it is opened once, later, by
scripts/evaluate.py after every decision is frozen.

Usage:
    uv run python scripts/train.py                 # all registered models
    uv run python scripts/train.py --models logreg
"""

from __future__ import annotations

import argparse
from typing import Any

from banknextaction.config import load_config
from banknextaction.data.fetch import fetch_dataset
from banknextaction.data.load import load_dataset
from banknextaction.data.split import apply_split, chronological_boundaries
from banknextaction.evaluation.metrics import choose_threshold_f1, evaluate
from banknextaction.models import available_models, create_model
from banknextaction.utils.io import write_json
from banknextaction.utils.logging import configure_logging, get_logger

log = get_logger("banknextaction.scripts.train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(available_models()))
    args = parser.parse_args()

    configure_logging()
    cfg = load_config()
    raw_file = cfg.path(cfg.data.raw_file)
    fetch_dataset(cfg.data, raw_file)  # verifies the SHA-256 before using the data
    features, target = load_dataset(raw_file, cfg.data, cfg.features)
    bounds = chronological_boundaries(len(features), cfg.split.train_frac, cfg.split.val_frac)
    parts = apply_split(features, target, bounds)
    x_train, y_train = parts["train"]
    x_val, y_val = parts["val"]

    models_dir = cfg.path(cfg.artifacts.models_dir)
    reports_dir = cfg.path(cfg.artifacts.reports_dir)
    for name in args.models:
        model = create_model(name, cfg)
        model.fit(x_train, y_train, x_val, y_val)
        proba_val = model.predict_proba(x_val)
        choice = choose_threshold_f1(
            y_val, proba_val, cfg.threshold.grid_size, cfg.threshold.tie_break
        )
        val_report = evaluate("val", y_val, proba_val, choice.threshold)
        written = model.save(models_dir)

        report: dict[str, Any] = {
            "model": model.describe(),
            "threshold": vars(choice),
            "validation": val_report.to_dict(),
            "train_seconds": round(type(model).fit.last_elapsed_ms / 1000.0, 2),
            "artifacts": [p.relative_to(cfg.root).as_posix() for p in written],
        }
        if getattr(model, "search", None) is not None:
            report["search"] = model.search.to_dict()
        if getattr(model, "history", None) is not None:
            report["training"] = model.training_summary()
        write_json(reports_dir / f"{name}_validation.json", report)
        log.info(
            "%s: threshold=%.3f val F1=%.4f PR-AUC=%.4f ROC-AUC=%.4f (floor acc %.4f)",
            name,
            choice.threshold,
            val_report.f1,
            val_report.pr_auc,
            val_report.roc_auc,
            val_report.floor_accuracy_majority,
        )


if __name__ == "__main__":
    main()
