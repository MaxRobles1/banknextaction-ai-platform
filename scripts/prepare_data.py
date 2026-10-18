"""Data step: download and verify the raw CSV, write the manifest and the EDA figure.

Usage:
    uv run python scripts/prepare_data.py [--force-download]
"""

from __future__ import annotations

import argparse

from banknextaction.config import load_config
from banknextaction.data.fetch import fetch_dataset
from banknextaction.data.load import load_raw, make_target
from banknextaction.data.manifest import build_manifest
from banknextaction.data.split import chronological_boundaries
from banknextaction.evaluation.figures import plot_eda
from banknextaction.utils.io import write_json
from banknextaction.utils.logging import configure_logging, get_logger

log = get_logger("banknextaction.scripts.prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true", help="download even if present")
    args = parser.parse_args()

    configure_logging()
    cfg = load_config()
    raw_file = cfg.path(cfg.data.raw_file)

    fetch = fetch_dataset(cfg.data, raw_file, force=args.force_download)
    raw = load_raw(raw_file, cfg.data.separator)
    target = make_target(raw, cfg.data)
    bounds = chronological_boundaries(len(raw), cfg.split.train_frac, cfg.split.val_frac)

    manifest = build_manifest(cfg, fetch, raw, target, bounds)
    write_json(cfg.path(cfg.data.manifest_file), manifest)
    figure = plot_eda(raw, target, bounds, cfg.path(cfg.artifacts.figures_dir) / "01_eda.png")

    log.info("manifest: %s", cfg.data.manifest_file)
    log.info("figure:   %s", figure.relative_to(cfg.root))
    for name, stats in manifest["split"]["summary"].items():
        log.info(
            "%-5s rows=%6d prevalence=%.4f majority-class accuracy=%.4f",
            name,
            stats["rows"],
            stats["prevalence"],
            stats["majority_class_accuracy"],
        )


if __name__ == "__main__":
    main()
