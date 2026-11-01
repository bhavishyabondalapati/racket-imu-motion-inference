"""D3 stroke-type classification under leave-one-player-out (LOPO).

For each test player: the remaining players are split into a fit set and an inner
validation set (whole players). Models are trained on the fit set; the validation
players are used for early stopping (deep models) and temperature scaling (all
probabilistic models). The test player is only ever used for prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from racketimu import io
from racketimu.dataset import inner_validation_players, lopo_test_players, to_arrays
from racketimu.evaluate import apply_temperature, fit_temperature, probs_to_logits
from racketimu.features import handcrafted_features, length_features, mean_features
from racketimu.models import build_model
from racketimu.train import (
    apply_scaler,
    class_weights,
    fit_scaler,
    predict_logits,
    train_classifier,
)

CLASSES = list(io.STROKES)
N_CLASSES = len(CLASSES)
FEATURE_MODELS = ["rf_mean_only", "rf_length_only", "rf_primary", "rf_primary_heading"]
DEEP_MODELS = ["cnn", "gru"]


@dataclass
class Fold:
    test_player: str
    val_players: list[str]
    fit: np.ndarray  # boolean masks over samples
    val: np.ndarray
    test: np.ndarray


@dataclass
class Inputs:
    meta: pd.DataFrame
    y: np.ndarray
    x_primary: np.ndarray
    mask: np.ndarray
    features: dict[str, np.ndarray]


def prepare_inputs(windows: pd.DataFrame, cfg: dict) -> Inputs:
    """Arrays for the primary channels and the feature sets for every random forest."""
    primary, heading = cfg["channels"]["primary"], cfg["channels"]["heading"]
    x_p, mask, meta = to_arrays(windows, primary)
    x_ph, _, _ = to_arrays(windows, primary + heading)
    features = {
        "rf_mean_only": mean_features(x_p, mask, primary)[0],
        "rf_length_only": length_features(mask)[0],
        "rf_primary": handcrafted_features(x_p, mask, primary)[0],
        "rf_primary_heading": handcrafted_features(x_ph, mask, primary + heading)[0],
    }
    y = meta["stroke"].map({s: i for i, s in enumerate(CLASSES)}).to_numpy()
    return Inputs(meta=meta, y=y, x_primary=x_p, mask=mask, features=features)


def make_folds(meta: pd.DataFrame, cfg: dict) -> list[Fold]:
    folds = []
    for k, player in enumerate(lopo_test_players(meta, cfg["lopo"]["excluded_test_players"])):
        test = (meta["player"] == player).to_numpy()
        val_players = inner_validation_players(
            meta[~test], cfg["classify"]["inner_val_players"], seed=cfg["seed"] + k
        )
        val = meta["player"].isin(val_players).to_numpy()
        folds.append(Fold(player, val_players, fit=~test & ~val, val=val, test=test))
    return folds


def _calibrate(val_logits, y_val, test_logits) -> tuple[np.ndarray, np.ndarray, float]:
    t = fit_temperature(val_logits, y_val)
    return apply_temperature(test_logits, 1.0), apply_temperature(test_logits, t), t


def run_majority(inp: Inputs, fold: Fold) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict the most frequent training class; probabilities are the training class shares."""
    prior = np.bincount(inp.y[fold.fit], minlength=N_CLASSES) / fold.fit.sum()
    probs = np.tile(prior, (fold.test.sum(), 1))
    return probs, probs, 1.0


def run_forest(name: str, inp: Inputs, fold: Fold, cfg: dict, seed: int):
    feats = inp.features[name]
    rf = RandomForestClassifier(
        n_estimators=cfg["random_forest"]["n_estimators"],
        max_depth=cfg["random_forest"]["max_depth"],
        class_weight=cfg["classify"]["class_weight"],
        random_state=seed,
        n_jobs=-1,
    ).fit(feats[fold.fit], inp.y[fold.fit])

    def proba(rows):
        p = np.zeros((rows.sum(), N_CLASSES))
        p[:, rf.classes_] = rf.predict_proba(feats[rows])
        return probs_to_logits(p)

    return _calibrate(proba(fold.val), inp.y[fold.val], proba(fold.test))


def run_deep(name: str, inp: Inputs, fold: Fold, cfg: dict, seed: int):
    mean, std = fit_scaler(inp.x_primary[fold.fit], inp.mask[fold.fit])
    sets = {}
    for part in ("fit", "val", "test"):
        rows = getattr(fold, part)
        x = apply_scaler(inp.x_primary[rows], inp.mask[rows], mean, std)
        sets[part] = (x, inp.mask[rows], inp.y[rows])
    model = build_model(name, inp.x_primary.shape[2], N_CLASSES, cfg)
    weights = class_weights(inp.y[fold.fit], N_CLASSES)
    model, info = train_classifier(model, sets["fit"], sets["val"], weights, cfg["train"], seed)
    val_logits = predict_logits(model, *sets["val"][:2])
    raw, cal, t = _calibrate(val_logits, inp.y[fold.val], predict_logits(model, *sets["test"][:2]))
    return raw, cal, t, info


def _rows(inp, fold, model, seed, raw, cal, t, info=None) -> pd.DataFrame:
    meta = inp.meta[fold.test]
    df = pd.DataFrame({
        "model": model, "seed": seed, "test_player": fold.test_player,
        "val_players": ",".join(fold.val_players), "sample_id": meta["sample_id"].to_numpy(),
        "level": meta["level"].to_numpy(), "y_true": inp.y[fold.test], "temperature": t,
        "best_epoch": (info or {}).get("best_epoch", np.nan),
    })  # fmt: skip
    for k, c in enumerate(CLASSES):
        df[f"p_raw_{c}"], df[f"p_cal_{c}"] = raw[:, k], cal[:, k]
    df["y_pred"] = raw.argmax(axis=1)
    return df


def run_all(windows: pd.DataFrame, cfg: dict, log=print) -> tuple[pd.DataFrame, list[Fold]]:
    """Every model x seed x fold; returns all out-of-fold predictions in one table."""
    inp = prepare_inputs(windows, cfg)
    folds = make_folds(inp.meta, cfg)
    parts = []
    for i, fold in enumerate(folds, 1):
        log(f"fold {i}/{len(folds)}: test {fold.test_player}, validation {fold.val_players}")
        parts.append(_rows(inp, fold, "majority", 0, *run_majority(inp, fold)))
        for seed in cfg["seeds"]:
            for name in FEATURE_MODELS:
                parts.append(_rows(inp, fold, name, seed, *run_forest(name, inp, fold, cfg, seed)))
            for name in DEEP_MODELS:
                parts.append(_rows(inp, fold, name, seed, *run_deep(name, inp, fold, cfg, seed)))
    return pd.concat(parts, ignore_index=True), folds
