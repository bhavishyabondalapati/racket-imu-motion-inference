"""Tests for hash-checked downloads, using local file:// URLs (no network)."""

import hashlib
from pathlib import Path

import pytest

from racketimu.assets import download_verified, sha256_of


@pytest.fixture
def source(tmp_path: Path) -> tuple[str, str]:
    src = tmp_path / "src" / "model.task"
    src.parent.mkdir()
    src.write_bytes(b"synthetic model bytes")
    return src.as_uri(), hashlib.sha256(b"synthetic model bytes").hexdigest()


def test_download_with_matching_hash(tmp_path: Path, source):
    url, digest = source
    dest = download_verified(url, tmp_path / "models" / "m.task", digest)
    assert dest.read_bytes() == b"synthetic model bytes"
    assert sha256_of(dest) == digest


def test_hash_mismatch_leaves_no_file(tmp_path: Path, source):
    url, _ = source
    dest = tmp_path / "models" / "m.task"
    with pytest.raises(ValueError, match="mismatch"):
        download_verified(url, dest, "0" * 64)
    assert not dest.exists()
    assert not dest.with_name("m.task.part").exists()


def test_existing_valid_file_is_kept(tmp_path: Path, source):
    url, digest = source
    dest = download_verified(url, tmp_path / "m.task", digest)
    mtime = dest.stat().st_mtime_ns
    download_verified(url, dest, digest)
    assert dest.stat().st_mtime_ns == mtime


def test_existing_corrupt_file_is_reported(tmp_path: Path, source):
    url, digest = source
    dest = tmp_path / "m.task"
    dest.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="does not match"):
        download_verified(url, dest, digest)
