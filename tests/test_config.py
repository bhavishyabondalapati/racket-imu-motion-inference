"""Tests for config loading. Uses only the repo's default config and temp files."""

from pathlib import Path

import pytest
import yaml

from racketimu.config import REQUIRED_SECTIONS, deep_merge, load_config


def test_default_config_has_required_sections():
    cfg = load_config()
    for section in REQUIRED_SECTIONS:
        assert section in cfg


def test_window_is_one_second_by_default():
    seg = load_config()["segment"]
    assert seg["window_pre_s"] + seg["window_post_s"] == pytest.approx(1.0)


def test_overrides_replace_only_the_given_keys():
    cfg = load_config(overrides={"segment": {"window_pre_s": 0.8}})
    assert cfg["segment"]["window_pre_s"] == 0.8
    assert cfg["segment"]["window_post_s"] == 0.4  # untouched sibling


def test_yaml_file_is_layered_on_defaults(tmp_path: Path):
    extra = tmp_path / "exp.yaml"
    extra.write_text(yaml.safe_dump({"imu": {"resample_hz": 200}}))
    cfg = load_config(extra)
    assert cfg["imu"]["resample_hz"] == 200
    assert "sync" in cfg


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": [1, 2]}}
    override = {"a": {"b": 2}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 2, "c": [1, 2]}}
    assert base == {"a": {"b": 1, "c": [1, 2]}}
