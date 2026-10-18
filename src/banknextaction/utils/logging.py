"""Logging setup: one place configures handlers; modules only call get_logger(__name__)."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure the package root logger once (level from argument or BNA_LOG_LEVEL)."""
    root = logging.getLogger("banknextaction")
    resolved = (level or os.environ.get("BNA_LOG_LEVEL", "INFO")).upper()
    root.setLevel(resolved)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return the module logger; use as get_logger(__name__)."""
    return logging.getLogger(name)
