"""Tests for D5 fold construction, on synthetic metadata."""

import numpy as np
import pandas as pd

from racketimu.skill import group_folds

CFG = {"lopo": {"excluded_test_players": ["P06"]}, "seed": 0}


def meta() -> pd.DataFrame:
    players = ["N01", "N02", "N03", "N04", "P01", "P02", "P03", "P04"]
    rows = [(p, "novice" if p[0] == "N" else "pro", s) for p in players
            for s in ("BASIC", "PUSH", "TOPSPIN")]  # fmt: skip
    return pd.DataFrame(rows, columns=["player", "level", "stroke"])


def test_within_group_folds_stay_inside_the_group():
    m = meta()
    folds = group_folds(m, "novice", "novice", CFG)
    assert [f.test_player for f in folds] == ["N01", "N02", "N03", "N04"]
    for f in folds:
        used = m.loc[f.fit | f.val | f.test, "level"].unique().tolist()
        assert used == ["novice"]
        assert not (f.fit & f.test).any() and not (f.val & f.test).any()
        assert len(f.val_players) == 2


def test_cross_group_trains_on_one_level_and_tests_on_the_other():
    m = meta()
    (fold,) = group_folds(m, "pro", "novice", CFG)
    assert set(m.loc[fold.test, "level"]) == {"novice"}
    assert set(m.loc[fold.fit | fold.val, "level"]) == {"pro"}
    assert all(p.startswith("P") for p in fold.val_players)
    assert np.all(fold.fit | fold.val | fold.test)
