"""Tests for D4 quality baselines and scoring, on synthetic data."""

import numpy as np
import pandas as pd

from racketimu.classify import Fold, Inputs
from racketimu.quality import TARGETS, beats_level_baseline, run_baselines, score


def test_baselines_use_training_players_only():
    meta = pd.DataFrame({"stroke": ["BASIC", "PUSH", "BASIC", "PUSH"],
                         "level": ["novice", "novice", "pro", "pro"]})  # fmt: skip
    y = np.array([[50.0] * 6, [60.0] * 6, [90.0] * 6, [100.0] * 6])
    inp = Inputs(meta=meta, y=np.zeros(4), x_primary=None, mask=None, features={})
    fold = Fold("t", [], fit=np.array([1, 1, 1, 0], bool), val=np.zeros(4, bool),
                test=np.array([0, 0, 0, 1], bool))  # fmt: skip
    b = run_baselines(inp, y, fold)
    assert b["global_mean"][0, 0] == np.mean([50, 60, 90])  # test sample (100) never used
    assert b["class_mean"][0, 0] == 60.0  # PUSH in training: only sample 2
    assert b["level_mean"][0, 0] == 90.0  # pro in training: only sample 3
    assert b["level_mean"].shape == (1, len(TARGETS))


def test_score_handles_constant_predictions():
    df = pd.DataFrame({"true_C1": [1.0, 2.0, 3.0], "pred_C1": [2.0, 2.0, 2.0]})
    s = score(df, "C1")
    assert s["mae"] == np.mean([1, 0, 1]) and np.isnan(s["spearman"])


def test_beat_rule_requires_every_seed():
    scores = pd.DataFrame({
        "method": ["level_mean", "rf", "rf", "rf"], "seed": [0, 0, 1, 2],
        "target": ["C1"] * 4, "mae": [5.0, 4.0, 4.5, 5.1],
    })  # fmt: skip
    assert beats_level_baseline(scores, "rf") == {"C1": False}
    scores.loc[3, "mae"] = 4.9
    assert beats_level_baseline(scores, "rf") == {"C1": True}
