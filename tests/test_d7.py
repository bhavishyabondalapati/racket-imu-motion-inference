"""Tests for D7 error-analysis helpers, on synthetic predictions."""

import pandas as pd

from racketimu.errors import confident_errors, confusion_pairs, seed_averaged


def oof() -> pd.DataFrame:
    rows = []
    for seed, p in [(0, [0.9, 0.05, 0.05]), (1, [0.5, 0.1, 0.4])]:  # sample 1: true TOPSPIN
        rows.append({"model": "rf", "seed": seed, "sample_id": 1, "test_player": "N01",
                     "level": "novice", "y_true": 2, "p_raw_BASIC": p[0], "p_raw_PUSH": p[1],
                     "p_raw_TOPSPIN": p[2]})  # fmt: skip
        rows.append({"model": "rf", "seed": seed, "sample_id": 2, "test_player": "P01",
                     "level": "pro", "y_true": 1, "p_raw_BASIC": 0.1, "p_raw_PUSH": 0.8,
                     "p_raw_TOPSPIN": 0.1})  # fmt: skip
    return pd.DataFrame(rows)


def test_probabilities_are_averaged_over_seeds():
    df = seed_averaged(oof(), "rf").set_index("sample_id")
    assert df.loc[1, "confidence"] == 0.7 and df.loc[1, "pred"] == "BASIC"
    assert not df.loc[1, "correct"] and df.loc[2, "correct"]


def test_confident_errors_and_pairs():
    df = seed_averaged(oof(), "rf")
    assert confident_errors(df)["sample_id"].tolist() == [1]
    assert confusion_pairs(df).to_dict("records") == [
        {"true": "TOPSPIN", "pred": "BASIC", "errors": 1}
    ]
