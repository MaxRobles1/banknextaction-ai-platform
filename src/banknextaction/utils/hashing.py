"""File digests for data and artifact integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file, read in chunks so large files do not load into memory."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return file_digest(path, "sha256")


def md5_file(path: Path) -> str:
    """MD5 of a file (kept for the Clase 12 receipt check, not for security)."""
    return file_digest(path, "md5")
