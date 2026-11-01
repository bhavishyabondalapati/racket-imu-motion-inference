"""Tests for config loading. Uses only the repo's default config and temp files."""

from pathlib import Path

import pytest
import yaml

from racketimu.config import REQUIRED_SECTIONS, deep_merge, load_config


def test_default_config_has_required_sections():
    cfg = load_config()
    for section in REQUIRED_SECTIONS:
        assert section in cfg


def test_default_window_is_48_rows_with_no_resampling():
    cfg = load_config()
    assert cfg["window"]["n_rows"] == 48
    assert cfg["window"]["peak_position"] == 16
    assert "imu" not in cfg  # no resampling setting: the native rate is kept


def test_overrides_replace_only_the_given_keys():
    cfg = load_config(overrides={"window": {"n_rows": 64}})
    assert cfg["window"]["n_rows"] == 64
    assert cfg["window"]["center"] == "peak_gyro"  # untouched sibling


def test_yaml_file_is_layered_on_defaults(tmp_path: Path):
    extra = tmp_path / "exp.yaml"
    extra.write_text(yaml.safe_dump({"ablation": {"row_strides": [1, 3]}}))
    cfg = load_config(extra)
    assert cfg["ablation"]["row_strides"] == [1, 3]
    assert "dataset" in cfg


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": [1, 2]}}
    override = {"a": {"b": 2}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 2, "c": [1, 2]}}
    assert base == {"a": {"b": 1, "c": [1, 2]}}
