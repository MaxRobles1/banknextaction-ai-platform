"""The model receipt (models/modelo.json): what the API serves, with integrity hashes.

It records file hashes (SHA-256 and the MD5 used in the Clase 12 check), the threshold
chosen on validation, library versions and the input contract, so anyone can verify that
the artifact in the image is exactly the one that was evaluated.
"""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from banknextaction.utils.hashing import md5_file, sha256_file
from banknextaction.utils.io import read_json

RECEIPT_LIBRARIES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "torch", "joblib")


def artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    """Name, size and digests of one artifact file."""
    return {
        "file": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "md5": md5_file(path),
    }


def library_versions() -> dict[str, str]:
    """Versions that must match between training and serving."""
    versions = {"python": platform.python_version()}
    versions.update({lib: version(lib) for lib in RECEIPT_LIBRARIES})
    return versions


def verify_receipt(receipt_path: Path, root: Path) -> list[str]:
    """Return the artifacts whose SHA-256 does not match the receipt (empty list = OK)."""
    receipt = read_json(receipt_path)
    mismatched = []
    for model in receipt["models"].values():
        for artifact in model["artifacts"]:
            if sha256_file(root / artifact["file"]) != artifact["sha256"]:
                mismatched.append(artifact["file"])
    return mismatched


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
