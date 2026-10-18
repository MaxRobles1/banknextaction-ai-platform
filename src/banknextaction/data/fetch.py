"""Reproducible download of the raw dataset, verified by SHA-256.

The raw CSV is never edited: it is downloaded once, checked against the hash recorded in
the configuration, and every later step reads it without writing back.
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from banknextaction.config import DataConfig
from banknextaction.utils.hashing import sha256_file
from banknextaction.utils.io import atomic_write
from banknextaction.utils.logging import get_logger

log = get_logger(__name__)

_DOWNLOAD_TIMEOUT_S = 120


class DataIntegrityError(RuntimeError):
    """The raw file does not match the SHA-256 recorded in the configuration."""


@dataclass(frozen=True)
class FetchResult:
    path: Path
    sha256: str
    downloaded: bool
    retrieved_at: str


def verify_sha256(path: Path, expected: str) -> str:
    """Return the file SHA-256, raising DataIntegrityError if it differs from `expected`."""
    actual = sha256_file(path)
    if actual != expected:
        raise DataIntegrityError(
            f"{path.name}: SHA-256 {actual} does not match the expected {expected}"
        )
    return actual


def extract_nested_member(archive_bytes: bytes, members: tuple[str, ...], dest: Path) -> Path:
    """Follow a chain of zip members (zip inside zip) and write the last one to `dest`."""
    payload = archive_bytes
    for member in members[:-1]:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            payload = archive.read(member)
    with (
        zipfile.ZipFile(io.BytesIO(payload)) as archive,
        archive.open(members[-1]) as src,
        atomic_write(dest, mode="wb") as out,
    ):
        shutil.copyfileobj(src, out)
    return dest


def download_bytes(url: str, timeout: int = _DOWNLOAD_TIMEOUT_S) -> bytes:
    """Download a URL into memory (the archive is about 1 MB)."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fetch_dataset(cfg: DataConfig, raw_file: Path, force: bool = False) -> FetchResult:
    """Make sure the verified raw CSV exists at `raw_file`, downloading it if needed."""
    if raw_file.exists() and not force:
        sha = verify_sha256(raw_file, cfg.raw_sha256)
        log.info("raw file already present and verified: %s", raw_file)
        retrieved = datetime.fromtimestamp(raw_file.stat().st_mtime, tz=UTC)
        return FetchResult(raw_file, sha, downloaded=False, retrieved_at=retrieved.isoformat())

    log.info("downloading %s", cfg.download_url)
    archive = download_bytes(cfg.download_url)
    extract_nested_member(archive, cfg.archive_members, raw_file)
    try:
        sha = verify_sha256(raw_file, cfg.raw_sha256)
    except DataIntegrityError:
        raw_file.unlink(missing_ok=True)
        raise
    log.info("downloaded and verified %s (sha256=%s)", raw_file, sha[:12])
    return FetchResult(raw_file, sha, downloaded=True, retrieved_at=datetime.now(UTC).isoformat())
