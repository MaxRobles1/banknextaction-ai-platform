"""File helpers that never leave half-written artifacts behind."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def atomic_write(path: Path, mode: str = "w") -> Iterator[IO[Any]]:
    """Write to a temporary file in the same folder and move it into place on success.

    If the block raises, the temporary file is removed and any previous version of `path`
    stays untouched. Used for datasets, manifests and model artifacts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        encoding = None if "b" in mode else "utf-8"
        with os.fdopen(fd, mode, encoding=encoding) as handle:
            yield handle
        tmp_path.chmod(0o644)  # mkstemp creates 0600; artifacts must be readable (Docker, CI)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON document with stable formatting."""
    with atomic_write(path) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON document."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
