"""Tests for the public inference interface, with tiny models trained on synthetic strokes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from racketimu.dataset import add_angle_encodings, window_stroke
from racketimu.features import handcrafted_features
from racketimu.infer import INPUT_COLUMNS, QUALITY_TARGETS, InferenceResult, StrokeModels, infer

PRIMARY = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "roll_deg", "pitch_sin",
           "pitch_cos"]  # fmt: skip
WINDOW = {"n_rows": 16, "peak_position": 5}


def stroke(rng, n=12) -> np.ndarray:
    w = rng.normal(size=(n, len(INPUT_COLUMNS)))
    w[:, 2] += 9.81
    w[:, 6:] *= 40
    return w


@pytest.fixture(scope="module")
def models() -> StrokeModels:
    rng = np.random.default_rng(0)
    feats = []
    for _ in range(30):
        df = add_angle_encodings(
            pd.DataFrame(stroke(rng), columns=INPUT_COLUMNS).assign(yaw_deg=0.0)
        )
        values, mask, _ = window_stroke(df, WINDOW["n_rows"], PRIMARY, WINDOW["peak_position"])
        feats.append(handcrafted_features(values[None], mask[None], PRIMARY)[0][0])
    feats = np.array(feats)
    labels = np.array(["BASIC", "PUSH", "TOPSPIN"] * 10)
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(feats, labels)
    scores = rng.uniform(50, 100, 30)
    regs = {t: RandomForestRegressor(n_estimators=5, random_state=0).fit(feats, scores)
            for t in QUALITY_TARGETS}  # fmt: skip
    return StrokeModels(clf, regs, PRIMARY, WINDOW, 8.0, "synthetic test models")


def test_infer_returns_a_complete_result(models):
    result = infer(stroke(np.random.default_rng(1)), fs=8.0, models=models)
    assert isinstance(result, InferenceResult)
    assert result.stroke_type in {"BASIC", "PUSH", "TOPSPIN"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == max(result.probabilities.values())
    assert result.confidence_is_calibrated is False
    assert set(result.quality) == set(QUALITY_TARGETS)
    assert 0 < result.rows_used <= WINDOW["n_rows"]


def test_higher_rate_is_reduced_to_the_training_rate(models):
    w = stroke(np.random.default_rng(2), n=40)
    assert infer(w, fs=32.0, models=models).rows_used <= 10  # every 4th row kept


def test_low_rate_warns(models):
    with pytest.warns(UserWarning, match="below the training rate"):
        infer(stroke(np.random.default_rng(3)), fs=4.0, models=models)


@pytest.mark.parametrize("bad", [np.zeros((5, 7)), np.zeros((0, 8)), np.full((5, 8), np.nan)])
def test_invalid_input_is_rejected(models, bad):
    with pytest.raises(ValueError):
        infer(bad, fs=8.0, models=models)
