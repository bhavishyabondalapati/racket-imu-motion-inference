"""Tests for the dataset loader, on a synthetic copy of the file layout."""

from datetime import datetime

import numpy as np
import pytest

from conftest import encode_timestamp
from racketimu import io


def test_decode_timestamp_with_and_without_seconds():
    assert io.decode_timestamp(encode_timestamp("2019年9月17日10时43分20秒")) == (
        datetime(2019, 9, 17, 10, 43, 20),
        True,
    )
    assert io.decode_timestamp(encode_timestamp("2019年9月19日11时37分")) == (
        datetime(2019, 9, 19, 11, 37),
        False,
    )
    assert io.decode_timestamp("not a timestamp") == (None, False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BASIC ", "BASIC"),
        ("Basic ", "BASIC"),
        ("Top", "TOPSPIN"),
        ("TOP ", "TOPSPIN"),
        ("push", "PUSH"),
    ],
)
def test_stroke_spellings_are_normalized(raw, expected):
    assert io.normalize_stroke(raw) == expected


def test_unknown_stroke_is_rejected():
    with pytest.raises(ValueError):
        io.normalize_stroke("backhand")


@pytest.mark.parametrize(("raw", "expected"), [("NO1", "N01"), ("PO8", "P08"), ("#P01 ", "P01")])
def test_player_codes_are_normalized(raw, expected):
    assert io.normalize_player(raw) == expected


def test_inclusion_rules_and_reasons(fake_dataset, dataset_cfg):
    ix = io.load_dataset(fake_dataset, dataset_cfg).index.set_index("sample_id")
    assert ix["exclusion_reason"].fillna("kept").to_dict() == {
        1: "kept",
        2: io.REASON_ACC_GYRO_ZERO,
        3: io.REASON_EMPTY,
        4: io.REASON_REMOVED_BY_AUTHORS,
        5: "kept",
        6: io.REASON_CLEAN_CSV_ONLY,
    }
    assert ix["included"].sum() == 2


def test_labels_levels_scores_and_override(fake_dataset, dataset_cfg):
    ix = io.load_dataset(fake_dataset, dataset_cfg).index.set_index("sample_id")
    assert ix.loc[1, ["player", "level", "stroke"]].tolist() == ["P01", "pro", "BASIC"]
    assert ix.loc[3, "level"] == "novice"
    assert ix.loc[5, "player"] == "P05" and ix.loc[5, "player_overridden"]
    assert not ix.loc[1, "player_overridden"]
    assert ix.loc[1, "score_mean"] == pytest.approx(np.mean([90, 90, 90, 90, 80]))
    assert (ix.loc[[1, 2, 3, 5], "folder_stroke"] == ix.loc[[1, 2, 3, 5], "stroke"]).all()


def test_units_are_converted(fake_dataset, dataset_cfg):
    stroke = io.load_dataset(fake_dataset, dataset_cfg).strokes[1]
    assert stroke["acc_x"].iloc[0] == pytest.approx(9.81)  # 981 / 100 m/s^2
    assert stroke["gyro_x"].iloc[0] == pytest.approx(np.deg2rad(1.0))  # 16 raw = 1 deg/s
    assert stroke["mag_x"].iloc[0] == pytest.approx(40.0)  # 640 / 16 uT
    assert stroke["yaw_deg"].iloc[0] == pytest.approx(30.0)  # Euler angles stay in degrees
    assert stroke["time_has_seconds"].all()


def test_duplicate_rate_counts_repeated_rows(fake_dataset, dataset_cfg):
    ix = io.load_dataset(fake_dataset, dataset_cfg).index.set_index("sample_id")
    assert ix.loc[1, "n_rows"] == 7
    assert ix.loc[1, "duplicate_rate"] == pytest.approx(1 / 6)
    assert np.isnan(ix.loc[3, "duplicate_rate"])


def test_missing_dataset_gives_a_clear_error(tmp_path, dataset_cfg):
    with pytest.raises(FileNotFoundError, match="Download the dataset"):
        io.load_dataset(tmp_path / "nowhere", dataset_cfg)
