"""Final evaluation: freeze decisions on VALIDATION, then open the test split once.

Steps, in this order:
1. Load the trained artifacts and the thresholds chosen on validation (scripts/train.py).
2. Select the served model with validation metrics, latency and size only, and write
   reports/model_selection.json before any test prediction exists.
3. Score the test split once with each model at its own validation threshold.
4. Write the comparison table, Figure 3 and the receipt models/modelo.json.

Usage:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banknextaction.config import load_config
from banknextaction.data.fetch import fetch_dataset
from banknextaction.data.load import load_dataset
from banknextaction.data.split import apply_split, chronological_boundaries
from banknextaction.evaluation.figures import plot_results
from banknextaction.evaluation.metrics import evaluate
from banknextaction.evaluation.receipt import artifact_entry, library_versions, utc_now
from banknextaction.evaluation.selection import (
    Candidate,
    measure_latency_ms,
    select_served_model,
)
from banknextaction.models import available_models, model_class
from banknextaction.utils.io import read_json, write_json
from banknextaction.utils.logging import configure_logging, get_logger

log = get_logger("banknextaction.scripts.evaluate")

# Plausibility ranges for the API (wider than the observed data, which is also recorded).
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "age": (18, 100),
    "balance": (-100_000, 1_000_000),
    "pdays": (-1, 999),
    "previous": (0, 300),
}


def _comparison_markdown(rows: list[dict[str, Any]], selection_reason: str, served: str) -> str:
    header = (
        "| Model | Split | Threshold | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy "
        "| Floor accuracy | Floor PR-AUC |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['model']}{' (served)' if r['model'] == served else ''} | {r['split']} "
        f"| {r['threshold']:.3f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['f1']:.4f} "
        f"| {r['precision']:.4f} | {r['recall']:.4f} | {r['accuracy']:.4f} "
        f"| {r['floor_accuracy_majority']:.4f} | {r['floor_pr_auc_prevalence']:.4f} |"
        for r in rows
    ]
    return header + "\n".join(lines) + f"\n\nServed model: **{served}** — {selection_reason}.\n"


def main() -> None:
    configure_logging()
    cfg = load_config()
    raw_file = cfg.path(cfg.data.raw_file)
    fetch_dataset(cfg.data, raw_file)
    features, target = load_dataset(raw_file, cfg.data, cfg.features)
    bounds = chronological_boundaries(len(features), cfg.split.train_frac, cfg.split.val_frac)
    parts = apply_split(features, target, bounds)
    x_val = parts["val"][0]  # validation inputs only: used for latency measurement

    models_dir = cfg.path(cfg.artifacts.models_dir)
    reports_dir = cfg.path(cfg.artifacts.reports_dir)

    # 1-2. Everything below is decided without the test split.
    loaded, validation, candidates = {}, {}, []
    for name in available_models():
        report = read_json(reports_dir / f"{name}_validation.json")
        model = model_class(name).load(models_dir)
        artifacts = [cfg.root / f for f in report["artifacts"]]
        latency = measure_latency_ms(model, x_val)
        loaded[name] = model
        validation[name] = {**report, "latency": latency}
        candidates.append(
            Candidate(
                name=name,
                val_f1=report["validation"]["f1"],
                val_pr_auc=report["validation"]["pr_auc"],
                latency_ms_p50=latency["p50_ms"],
                artifact_bytes=sum(p.stat().st_size for p in artifacts),
            )
        )
    selection = select_served_model(candidates)
    write_json(
        reports_dir / "model_selection.json",
        {
            "decided_at": utc_now(),
            "uses": "validation metrics, latency and size only",
            "served": selection.served,
            "reason": selection.reason,
            "ranking": selection.ranking,
        },
    )
    log.info("served model: %s (%s)", selection.served, selection.reason)

    # 3. The test split is opened here, once, with every decision already frozen.
    x_test, y_test = parts["test"]
    rows, test_probas, per_model = [], {}, {}
    for name, model in loaded.items():
        threshold = validation[name]["threshold"]["threshold"]
        proba = model.predict_proba(x_test)
        test_probas[name] = proba
        test_report = evaluate("test", y_test, proba, threshold).to_dict()
        rows += [{"model": name, **validation[name]["validation"]}, {"model": name, **test_report}]
        per_model[name] = {"validation": validation[name]["validation"], "test": test_report}
        log.info(
            "%s test: F1=%.4f PR-AUC=%.4f ROC-AUC=%.4f acc=%.4f (floor %.4f)",
            name,
            test_report["f1"],
            test_report["pr_auc"],
            test_report["roc_auc"],
            test_report["accuracy"],
            test_report["floor_accuracy_majority"],
        )

    served = selection.served
    served_threshold = validation[served]["threshold"]["threshold"]
    write_json(
        reports_dir / "test_evaluation.json",
        {"evaluated_at": utc_now(), "served": served, "models": per_model},
    )
    (reports_dir / "model_comparison.md").write_text(
        _comparison_markdown(rows, selection.reason, served), encoding="utf-8"
    )
    plot_results(
        np.asarray(y_test),
        test_probas,
        served,
        served_threshold,
        cfg.path(cfg.artifacts.figures_dir) / "03_test_results.png",
    )

    # 4. Receipt: what the API serves, with hashes and the input contract.
    manifest = read_json(cfg.path(cfg.data.manifest_file))
    train_x = parts["train"][0]
    encoder = loaded["logreg"].pipeline.named_steps["preprocess"].named_transformers_["cat"]
    categories = {
        col: [str(v) for v in values]
        for col, values in zip(cfg.features.categorical, encoder.categories_, strict=True)
    }
    receipt = {
        "project": cfg.project.name,
        "model_version": cfg.project.model_version,
        "generated_at": utc_now(),
        "served_model": served,
        "decision_threshold": served_threshold,
        "threshold_rule": validation[served]["threshold"]["criterion"],
        "selection_reason": selection.reason,
        "models": {
            name: {
                "artifacts": [
                    {
                        **artifact_entry(cfg.root / f, cfg.root),
                        "created_at": _mtime(cfg.root / f),
                    }
                    for f in validation[name]["artifacts"]
                ],
                "threshold": validation[name]["threshold"]["threshold"],
                "metrics": per_model[name],
                "latency_single_row": validation[name]["latency"],
                "description": validation[name]["model"],
            }
            for name in loaded
        },
        "input_schema": {
            "order": list(cfg.features.inputs),
            "numeric": {
                col: {
                    "allowed_range": list(NUMERIC_RANGES[col]),
                    "observed_train_range": [int(train_x[col].min()), int(train_x[col].max())],
                }
                for col in cfg.features.numeric
            },
            "categorical": categories,
        },
        "data": {
            "dataset_sha256": manifest["dataset"]["sha256"],
            "data_code_sha256": manifest["reproducibility"]["data_code_sha256"],
            "split_boundaries": manifest["split"]["boundaries"],
        },
        "versions": library_versions(),
    }
    write_json(cfg.path(cfg.artifacts.receipt_file), receipt)
    log.info("receipt written: %s", cfg.artifacts.receipt_file)


def _mtime(path: Any) -> str:
    from datetime import UTC, datetime  # noqa: PLC0415

    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
