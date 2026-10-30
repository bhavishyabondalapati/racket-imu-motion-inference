"""Download external model files and verify them against a pinned SHA-256 hash."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks to keep memory flat."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, dest: Path, expected_sha256: str) -> Path:
    """Download `url` to `dest` unless a file with the expected hash is already there.

    The download goes to a temporary `.part` file first and is only renamed to `dest`
    after its hash matches, so a failed or tampered download never looks valid.
    Raises ValueError on a hash mismatch.
    """
    expected = expected_sha256.lower()
    if dest.exists():
        if sha256_of(dest) == expected:
            return dest
        raise ValueError(
            f"{dest} exists but its SHA-256 does not match the config. "
            "Delete it by hand and rerun to download a fresh copy."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, part)
        actual = sha256_of(part)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")
        part.rename(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest
