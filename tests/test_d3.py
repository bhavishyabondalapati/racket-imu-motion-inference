"""Tests for D3 building blocks, on synthetic data only."""

import numpy as np
import pandas as pd
import pytest
import torch

from racketimu.dataset import inner_validation_players, lopo_test_players
from racketimu.evaluate import apply_temperature, ece, fit_temperature, group_metrics
from racketimu.features import handcrafted_features, length_features, mean_features
from racketimu.models import StrokeCNN, StrokeGRU
from racketimu.train import apply_scaler, class_weights, fit_scaler, train_classifier

PRIMARY = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "roll_deg", "pitch_sin",
           "pitch_cos"]  # fmt: skip


def synthetic(n=6, t=10, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, t, len(PRIMARY))).astype(np.float32)
    pitch = rng.uniform(-np.pi, np.pi, size=(n, t))
    x[:, :, 7], x[:, :, 8] = np.sin(pitch), np.cos(pitch)
    mask = np.ones((n, t), dtype=bool)
    mask[0, :3] = False  # front padding
    if n > 1:
        mask[1, 7:] = False  # back padding
    x[~mask] = 0.0
    return x, mask


def test_features_ignore_padded_rows():
    x, mask = synthetic()
    feats, names = handcrafted_features(x, mask, PRIMARY)
    assert feats.shape == (6, len(names)) and np.isfinite(feats).all()
    x2 = x.copy()
    x2[~mask] = 99.0  # changing padding must not change any feature
    np.testing.assert_allclose(handcrafted_features(x2, mask, PRIMARY)[0], feats)
    means, _ = mean_features(x, mask, PRIMARY)
    np.testing.assert_allclose(means[0], x[0, 3:].mean(axis=0), rtol=1e-5)
    assert length_features(mask)[0][:, 0].tolist() == [7, 7, 10, 10, 10, 10]


def test_pitch_change_handles_wraparound():
    x, mask = synthetic(n=1, t=2)
    mask[:] = True
    for k, deg in enumerate([179.0, -179.0]):
        x[0, k, 7], x[0, k, 8] = np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))
    feats, names = handcrafted_features(x, mask, PRIMARY)
    assert feats[0, names.index("pitch_change")] == pytest.approx(2.0, abs=1e-3)
    assert feats[0, names.index("pitch_path")] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("model", [StrokeCNN(9, 3, [8, 8], 3, 0.0), StrokeGRU(9, 3, 8, 0.0)])
def test_models_ignore_padding_content(model):
    x, mask = synthetic()
    model.eval()
    with torch.no_grad():
        a = model(torch.from_numpy(x), torch.from_numpy(mask))
        if isinstance(model, StrokeGRU):  # front padding feeds the GRU state; check back padding
            x2 = x.copy()
            x2[1, 7:] = 5.0
        else:  # the CNN sees neighbours at the edges; check rows far from real data
            x2 = x.copy()
            x2[0, 0] = 5.0
        b = model(torch.from_numpy(x2), torch.from_numpy(mask))
    assert a.shape == (6, 3)
    torch.testing.assert_close(
        a[1:] if isinstance(model, StrokeCNN) else a[1],
        b[1:] if isinstance(model, StrokeCNN) else b[1],
    )


def test_scaler_uses_real_rows_and_rezeroes_padding():
    x, mask = synthetic()
    mean, std = fit_scaler(x, mask)
    np.testing.assert_allclose(mean, x[mask].mean(axis=0), rtol=1e-5)
    z = apply_scaler(x, mask, mean, std)
    assert (z[~mask] == 0).all()
    np.testing.assert_allclose(z[mask].mean(axis=0), 0, atol=1e-5)


def test_class_weights_are_balanced():
    np.testing.assert_allclose(class_weights(np.array([0, 0, 0, 1]), 3), [4 / 9, 4 / 3, 0.0])


def test_training_runs_and_restores_best_epoch():
    x, mask = synthetic(n=12)
    y = np.array([0, 1, 2] * 4)
    cfg = {"device": "cpu", "learning_rate": 0.01, "batch_size": 4, "max_epochs": 5,
           "early_stopping_patience": 2}  # fmt: skip
    model, info = train_classifier(StrokeCNN(9, 3, [4], 3, 0.0), (x, mask, y), (x, mask, y),
                                   np.ones(3), cfg, seed=0)  # fmt: skip
    assert 0 <= info["best_epoch"] < info["epochs_run"] <= 5


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y] += 1.0
    logits *= 4.0  # overconfident
    t = fit_temperature(logits, y)
    assert t > 2.0
    assert ece(apply_temperature(logits, t), y) < ece(apply_temperature(logits, 1.0), y)


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array([[0.75, 0.25]] * 4)
    assert ece(probs, np.array([0, 0, 0, 1]), n_bins=10) == pytest.approx(0.0)


def test_group_metrics_averages_only_present_classes():
    df = pd.DataFrame({"g": ["a"] * 4, "y_true": [0, 0, 0, 0], "y_pred": [0, 0, 1, 1]})
    row = group_metrics(df, "g", ["A", "B"]).iloc[0]
    assert row["classes"] == "A" and row["accuracy"] == 0.5
    assert row["macro_f1_present"] == pytest.approx(2 / 3)


def test_lopo_and_inner_validation_never_overlap_with_test():
    meta = pd.DataFrame({
        "player": ["N01"] * 3 + ["N02"] * 3 + ["P01"] * 3 + ["P02"] * 3 + ["P03"],
        "level": ["novice"] * 6 + ["pro"] * 7,
        "stroke": ["BASIC", "PUSH", "TOPSPIN"] * 4 + ["BASIC"],
    })  # fmt: skip
    assert lopo_test_players(meta, ["P03"]) == ["N01", "N02", "P01", "P02"]
    for test in lopo_test_players(meta, ["P03"]):
        val = inner_validation_players(meta[meta["player"] != test], {"novice": 1, "pro": 1}, 0)
        assert test not in val and len(val) == 2
        assert "P03" not in val  # only players with all three classes are preferred
