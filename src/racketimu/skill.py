"""D5: skill level (novice vs pro) and cross-group generalization.

Part 1: classify each stroke as novice or pro under the D3 LOPO folds. Each test
player has a single level, so per-fold F1 is meaningless; results are pooled
out-of-fold metrics plus per-player accuracy.

Part 2: train the stroke-type RF and the quality RF on one level and test on the
other, compared with LOPO within that level. Validation players always come from
the training group. As in D3/D4, the forests are fitted on the fit set only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from racketimu.classify import Fold, Inputs, make_folds
from racketimu.dataset import inner_validation_players
from racketimu.models import build_model
from racketimu.quality import TARGETS, run_rf_regressor, score, targets
from racketimu.seed import set_seed
from racketimu.train import (
    apply_scaler,
    class_weights,
    fit_scaler,
    predict_logits,
    train_classifier,
)

LEVELS = ["novice", "pro"]


def level_labels(inp: Inputs) -> np.ndarray:
    return (inp.meta["level"] == "pro").to_numpy().astype(int)


def _forest(cfg: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg["random_forest"]["n_estimators"],
        max_depth=cfg["random_forest"]["max_depth"],
        class_weight=cfg["classify"]["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )


def predict_forest(inp, y, fold, cfg, seed, n_classes) -> np.ndarray:
    """Class probabilities [n_test, n_classes] from the handcrafted-feature RF."""
    feats = inp.features["rf_primary"]
    rf = _forest(cfg, seed).fit(feats[fold.fit], y[fold.fit])
    p = np.zeros((fold.test.sum(), n_classes))
    p[:, rf.classes_] = rf.predict_proba(feats[fold.test])
    return p


def predict_deep_level(name, inp, y, fold, cfg, seed) -> np.ndarray:
    """Deep novice/pro classifier; same training recipe as D3 with 2 classes."""
    mean, std = fit_scaler(inp.x_primary[fold.fit], inp.mask[fold.fit])
    sets = {}
    for part in ("fit", "val", "test"):
        rows = getattr(fold, part)
        sets[part] = (
            apply_scaler(inp.x_primary[rows], inp.mask[rows], mean, std),
            inp.mask[rows],
            y[rows],
        )
    set_seed(seed)
    model = build_model(name, inp.x_primary.shape[2], 2, cfg)
    model, _ = train_classifier(
        model, sets["fit"], sets["val"], class_weights(y[fold.fit], 2), cfg["train"], seed
    )
    return predict_logits(model, *sets["test"][:2]).argmax(axis=1)


def run_level_classification(inp: Inputs, cfg: dict) -> pd.DataFrame:
    y = level_labels(inp)
    deep = cfg["ablation"]["deep_model"]
    rows = []
    for fold in make_folds(inp.meta, cfg):
        meta = inp.meta[fold.test]
        base = {
            "test_player": fold.test_player,
            "sample_id": meta["sample_id"].to_numpy(),
            "stroke": meta["stroke"].to_numpy(),
            "y_true": y[fold.test],
        }
        majority = int(np.bincount(y[fold.fit], minlength=2).argmax())
        rows.append(pd.DataFrame({**base, "model": "majority", "seed": 0, "y_pred": majority}))
        for seed in cfg["seeds"]:
            p = predict_forest(inp, y, fold, cfg, seed, 2).argmax(axis=1)
            rows.append(pd.DataFrame({**base, "model": "rf", "seed": seed, "y_pred": p}))
            p = predict_deep_level(deep, inp, y, fold, cfg, seed)
            rows.append(pd.DataFrame({**base, "model": deep, "seed": seed, "y_pred": p}))
    return pd.concat(rows, ignore_index=True)


def level_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), g in oof.groupby(["model", "seed"], sort=False):
        rows.append(
            {
                "model": model,
                "seed": seed,
                "accuracy": (g["y_true"] == g["y_pred"]).mean(),
                "balanced_accuracy": balanced_accuracy_score(g["y_true"], g["y_pred"]),
                "macro_f1": f1_score(
                    g["y_true"], g["y_pred"], labels=[0, 1], average="macro", zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows)


def group_folds(meta: pd.DataFrame, train_level: str, test_level: str, cfg: dict) -> list[Fold]:
    """Within-group LOPO (train_level == test_level) or one cross-group fold.

    Validation players: 2 from the training group, preferring players with all classes.
    """
    excluded = set(cfg["lopo"]["excluded_test_players"])
    if train_level == test_level:
        players = sorted(
            p
            for p in meta.loc[meta["level"] == train_level, "player"].unique()
            if p not in excluded
        )
        folds = []
        for k, player in enumerate(players):
            test = (meta["player"] == player).to_numpy()
            pool = meta[(meta["level"] == train_level) & ~test]
            folds.append(_fold(meta, player, test, pool, train_level, cfg["seed"] + k))
        return folds
    test = (meta["level"] == test_level).to_numpy()
    pool = meta[meta["level"] == train_level]
    return [_fold(meta, f"all {test_level}s", test, pool, train_level, cfg["seed"])]


def _fold(meta, name, test, pool, level, seed) -> Fold:
    val_players = inner_validation_players(pool, {level: 2}, seed)
    val = meta["player"].isin(val_players).to_numpy()
    fit = meta["player"].isin(pool["player"].unique()).to_numpy() & ~val & ~test
    return Fold(name, val_players, fit=fit, val=val, test=test)


def run_cross_group(inp: Inputs, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stroke-type and quality RF out-of-fold predictions for the four train/test settings."""
    y_type = inp.y
    y_score = targets(inp)
    type_rows, score_rows = [], []
    for train_level in LEVELS:
        for test_level in LEVELS:
            setting = f"train {train_level}s -> test {test_level}s"
            mode = "within-group LOPO" if train_level == test_level else "cross-group"
            for fold in group_folds(inp.meta, train_level, test_level, cfg):
                for seed in cfg["seeds"]:
                    tag = {
                        "setting": setting,
                        "mode": mode,
                        "test_level": test_level,
                        "seed": seed,
                        "fold": fold.test_player,
                    }
                    pred = predict_forest(inp, y_type, fold, cfg, seed, 3).argmax(axis=1)
                    type_rows.append(
                        pd.DataFrame({**tag, "y_true": y_type[fold.test], "y_pred": pred})
                    )
                    preds = run_rf_regressor(inp, y_score, fold, cfg, seed)
                    df = pd.DataFrame(
                        tag | {"sample_id": inp.meta["sample_id"].to_numpy()[fold.test]}
                    )
                    for k, t in enumerate(TARGETS):
                        df[f"true_{t}"], df[f"pred_{t}"] = y_score[fold.test, k], preds[:, k]
                    score_rows.append(df)
    return pd.concat(type_rows, ignore_index=True), pd.concat(score_rows, ignore_index=True)


def cross_group_summary(type_oof: pd.DataFrame, score_oof: pd.DataFrame) -> pd.DataFrame:
    """Per setting: stroke-type macro-F1 and score_mean MAE/Spearman, mean ± std over seeds."""
    rows = []
    for (setting, seed), g in type_oof.groupby(["setting", "seed"], sort=False):
        s = score_oof[(score_oof["setting"] == setting) & (score_oof["seed"] == seed)]
        present = sorted(g["y_true"].unique())
        rows.append(
            {
                "setting": setting,
                "mode": g["mode"].iloc[0],
                "test_level": g["test_level"].iloc[0],
                "seed": seed,
                "n_test": len(g),
                "type_macro_f1": f1_score(
                    g["y_true"], g["y_pred"], labels=present, average="macro", zero_division=0
                ),
                **{f"{k}_score_mean": v for k, v in score(s, "score_mean").items()},
            }
        )
    return pd.DataFrame(rows)
