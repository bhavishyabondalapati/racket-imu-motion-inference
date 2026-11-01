"""Tests for the dataset card builder, on the synthetic dataset from conftest."""

import pandas as pd

from racketimu import io
from racketimu.datacard import build_dataset_card, id_ranges, md_table, rows_per_whole_second


def test_id_ranges_compresses_runs():
    assert id_ranges([7, 1, 2, 3, 9, 10]) == "1-3, 7, 9-10"
    assert id_ranges([]) == "none"


def test_md_table_renders_header_and_rows():
    text = md_table(pd.DataFrame({"a": [1], "b": ["x"]}), index=False)
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |"]


def test_rows_per_whole_second_skips_partial_seconds():
    t = pd.to_datetime(["2019-09-17 10:00:00"] * 2 + ["2019-09-17 10:00:01"] * 9
                       + ["2019-09-17 10:00:02"] * 7 + ["2019-09-17 10:00:03"] * 3)  # fmt: skip
    stroke = pd.DataFrame({"time": t, "time_has_seconds": True})
    minute_only = stroke.assign(time_has_seconds=False)
    assert rows_per_whole_second([stroke, minute_only]).tolist() == [9, 7]


def test_card_reconciles_counts_and_lists_every_exclusion(fake_dataset, dataset_cfg):
    ds = io.load_dataset(fake_dataset, dataset_cfg)
    cfg = {"dataset": {**dataset_cfg, "root": "fake"}, "lopo": {"excluded_test_players": []}}
    card = build_dataset_card(ds, io.DatasetFiles(fake_dataset), cfg)
    assert "| **Used in this project** | 2 |" in card
    for reason in [io.REASON_REMOVED_BY_AUTHORS, io.REASON_EMPTY, io.REASON_ACC_GYRO_ZERO]:
        assert reason[1:] in card
    assert "Agreement between coaches cannot be computed" in card
    assert "**Observed rows per whole second:** mean 3.00" in card  # sample 5, second 21
