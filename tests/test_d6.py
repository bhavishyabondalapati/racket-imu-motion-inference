"""Tests for the D6 ablation conditions, on synthetic windows."""

import numpy as np
import pandas as pd

from racketimu.ablation import Condition, conditions, select_rows

CFG = {
    "window": {"n_rows": 48, "peak_position": 16},
    "ablation": {"row_strides": [1, 2, 4], "window_fractions": [1.0, 0.75, 0.5]},
    "channels": {"primary": ["acc_x"], "heading": ["mag_x"]},
}


def windows() -> pd.DataFrame:
    return pd.DataFrame({"sample_id": 1, "pos": np.arange(48) - 16, "mask": True})


def test_stride_keeps_the_peak_and_every_kth_row():
    rows = select_rows(windows(), Condition("s4", "rate", "primary", stride=4), CFG)
    assert 0 in rows["pos"].tolist()
    assert len(rows) == 12 and (rows["pos"] % 4 == 0).all()


def test_shorter_windows_keep_the_peak_at_the_same_relative_position():
    for frac, n, first in [(1.0, 48, -16), (0.75, 36, -12), (0.5, 24, -8)]:
        rows = select_rows(
            windows(), Condition("w", "window", "primary", window_fraction=frac), CFG
        )
        assert len(rows) == n and rows["pos"].min() == first


def test_condition_grid_matches_the_plan():
    names = [c.name for c in conditions(CFG)]
    assert names == ["full", "stride_2", "stride_4", "acc_only", "gyro_only", "acc_gyro",
                     "primary_heading", "window_75", "window_50", "motion_only"]  # fmt: skip
    deep = [c.name for c in conditions(CFG) if c.deep]
    assert deep == ["full", "stride_4", "motion_only"]
