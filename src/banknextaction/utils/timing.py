"""Timing decorator used to measure real latency (training steps, inference)."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from banknextaction.utils.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")

log = get_logger(__name__)


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Log the wall-clock duration of each call and keep the last one in `last_elapsed_ms`.

    The API reports this value as `inference_time_ms`, so the number shown to users is
    measured, not assumed.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            wrapper.last_elapsed_ms = elapsed_ms  # type: ignore[attr-defined]
            log.debug("%s took %.2f ms", func.__qualname__, elapsed_ms)

    wrapper.last_elapsed_ms = float("nan")  # type: ignore[attr-defined]
    return wrapper
