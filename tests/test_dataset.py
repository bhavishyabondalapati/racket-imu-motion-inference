"""Tests for D2 windowing, on synthetic strokes."""

import numpy as np
import pandas as pd
import pytest

from racketimu import io
from racketimu.dataset import (
    CHANNELS,
    add_angle_encodings,
    build_windows,
    to_arrays,
    window_bounds,
    window_stroke,
)


def make_stroke(n: int, peak: int) -> pd.DataFrame:
    """A stroke whose gyro_x is the row number, except a spike at `peak`."""
    data = {c: np.zeros(n) for c in io.CHANNELS}
    data["gyro_x"] = np.arange(n, dtype=float)
    data["gyro_x"][peak] = 1000.0
    data["pitch_deg"] = np.full(n, 179.0)
    return add_angle_encodings(pd.DataFrame(data))


def test_peak_lands_in_the_middle_of_a_padded_window():
    values, mask, info = window_stroke(make_stroke(10, peak=3), n_rows=8)
    gyro_x = values[:, CHANNELS.index("gyro_x")]
    assert gyro_x[4] == 1000.0  # n_rows // 2
    assert mask.tolist() == [False, True, True, True, True, True, True, True]
    assert gyro_x[0] == 0.0 and info == {"peak_row": 3, "rows_cropped": 3}


def test_long_stroke_is_cropped_around_the_peak():
    values, mask, info = window_stroke(make_stroke(50, peak=30), n_rows=8)
    gyro_x = values[:, CHANNELS.index("gyro_x")]
    assert mask.all()
    assert gyro_x.tolist() == [26, 27, 28, 29, 1000, 31, 32, 33]
    assert info["rows_cropped"] == 42


def test_peak_position_moves_the_peak():
    values, mask, _ = window_stroke(make_stroke(10, peak=3), n_rows=8, peak_position=2)
    gyro_x = values[:, CHANNELS.index("gyro_x")]
    assert gyro_x[2] == 1000.0
    assert gyro_x.tolist()[:3] == [1, 2, 1000] and mask.all()


def test_window_bounds_at_both_edges():
    assert window_bounds(n_stroke=5, peak=0, n_rows=8) == (0, 4, 4)
    assert window_bounds(n_stroke=5, peak=4, n_rows=8) == (0, 5, 0)


def test_empty_stroke_is_rejected():
    with pytest.raises(ValueError):
        window_stroke(make_stroke(1, 0).iloc[:0], n_rows=8)


def test_pitch_near_180_is_continuous_in_sin_cos():
    stroke = pd.DataFrame({c: [0.0, 0.0] for c in io.CHANNELS}).assign(pitch_deg=[179.0, -179.0])
    enc = add_angle_encodings(stroke)
    assert abs(enc["pitch_deg"].diff().iloc[1]) == pytest.approx(358.0)  # raw angle jumps
    step = np.hypot(enc["pitch_sin"].diff().iloc[1], enc["pitch_cos"].diff().iloc[1])
    assert step == pytest.approx(np.deg2rad(2.0), abs=1e-3)  # encoded step is small


def test_build_windows_and_arrays_on_fake_dataset(fake_dataset, dataset_cfg):
    ds = io.load_dataset(fake_dataset, dataset_cfg)
    windows = build_windows(ds, {"n_rows": 8, "center": "peak_gyro"})
    assert sorted(windows["sample_id"].unique()) == [1, 5]  # included samples only
    x, mask, meta = to_arrays(windows, ["gyro_x", "roll_deg"])
    assert x.shape == (2, 8, 2) and mask.shape == (2, 8)
    assert meta["stroke"].tolist() == ["BASIC", "PUSH"]
    assert (x[~mask] == 0).all()
