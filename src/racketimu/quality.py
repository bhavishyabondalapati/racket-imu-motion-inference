"""D4 stroke-quality estimation: predict the coach scores C1-C5 and their mean.

Same LOPO folds and inner validation players as D3. Baselines are fitted on all
training players (fit + validation), which makes them as strong as possible; the
models are fitted on the fit set only, with the validation players used for early
stopping (GRU), exactly as in D3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from racketimu import io
from racketimu.classify import Fold, Inputs, make_folds, prepare_inputs
from racketimu.models import build_model
from racketimu.seed import set_seed
from racketimu.train import apply_scaler, fit_scaler, predict_logits, train_regressor

TARGETS = [*io.CRITERIA, "score_mean"]
BASELINES = ["global_mean", "class_mean", "level_mean"]


def targets(inp: Inputs) -> np.ndarray:
    return inp.meta[TARGETS].to_numpy(dtype=np.float64)


def run_baselines(inp: Inputs, y: np.ndarray, fold: Fold) -> dict[str, np.ndarray]:
    """Global training mean; training mean of the test sample's (true) stroke class;
    training mean of the test player's level."""
    train = fold.fit | fold.val
    meta = inp.meta
    test_meta = meta[fold.test]
    out = {"global_mean": np.tile(y[train].mean(axis=0), (fold.test.sum(), 1))}
    for name, col in [("class_mean", "stroke"), ("level_mean", "level")]:
        means = (
            pd.DataFrame(y[train], columns=TARGETS).groupby(meta.loc[train, col].to_numpy()).mean()
        )
        out[name] = means.reindex(test_meta[col].to_numpy()).to_numpy()
    return out


def run_rf_regressor(inp: Inputs, y: np.ndarray, fold: Fold, cfg: dict, seed: int) -> np.ndarray:
    """One random-forest regressor per target on the handcrafted primary features."""
    feats = inp.features["rf_primary"]
    preds = []
    for k in range(len(TARGETS)):
        rf = RandomForestRegressor(
            n_estimators=cfg["random_forest"]["n_estimators"],
            max_depth=cfg["random_forest"]["max_depth"],
            random_state=seed,
            n_jobs=-1,
        ).fit(feats[fold.fit], y[fold.fit, k])
        preds.append(rf.predict(feats[fold.test]))
    return np.column_stack(preds)


def run_deep_regressor(
    name: str, inp: Inputs, y: np.ndarray, fold: Fold, cfg: dict, seed: int
) -> tuple[np.ndarray, dict]:
    """The D3 deep model with one output per target; targets standardized on the fit set."""
    mean, std = fit_scaler(inp.x_primary[fold.fit], inp.mask[fold.fit])
    y_mean, y_std = y[fold.fit].mean(axis=0), y[fold.fit].std(axis=0)
    y_std = np.where(y_std > 0, y_std, 1.0)
    sets = {}
    for part in ("fit", "val", "test"):
        rows = getattr(fold, part)
        x = apply_scaler(inp.x_primary[rows], inp.mask[rows], mean, std)
        sets[part] = (x, inp.mask[rows], ((y[rows] - y_mean) / y_std).astype(np.float32))
    set_seed(seed)  # before building, so the initial weights depend only on the seed
    model = build_model(name, inp.x_primary.shape[2], len(TARGETS), cfg)
    model, info = train_regressor(model, sets["fit"], sets["val"], cfg["train"], seed)
    return predict_logits(model, *sets["test"][:2]) * y_std + y_mean, info


def _rows(inp, y, fold, method, seed, preds) -> pd.DataFrame:
    meta = inp.meta[fold.test]
    df = pd.DataFrame({
        "method": method, "seed": seed, "test_player": fold.test_player,
        "sample_id": meta["sample_id"].to_numpy(), "level": meta["level"].to_numpy(),
        "stroke": meta["stroke"].to_numpy(),
    })  # fmt: skip
    for k, t in enumerate(TARGETS):
        df[f"true_{t}"], df[f"pred_{t}"] = y[fold.test, k], preds[:, k]
    return df


def run_all(windows: pd.DataFrame, cfg: dict, log=print) -> pd.DataFrame:
    inp = prepare_inputs(windows, cfg)
    y = targets(inp)
    deep = cfg["ablation"]["deep_model"]
    parts = []
    for i, fold in enumerate(make_folds(inp.meta, cfg), 1):
        log(f"fold {i}: test {fold.test_player}")
        for name, preds in run_baselines(inp, y, fold).items():
            parts.append(_rows(inp, y, fold, name, 0, preds))
        for seed in cfg["seeds"]:
            parts.append(_rows(inp, y, fold, "rf", seed, run_rf_regressor(inp, y, fold, cfg, seed)))
            preds, _ = run_deep_regressor(deep, inp, y, fold, cfg, seed)
            parts.append(_rows(inp, y, fold, deep, seed, preds))
    return pd.concat(parts, ignore_index=True)


def score(df: pd.DataFrame, target: str) -> dict[str, float]:
    """MAE and Spearman correlation for one target (Spearman is NaN if either side is constant)."""
    t, p = df[f"true_{target}"].to_numpy(), df[f"pred_{target}"].to_numpy()
    rho = np.nan if np.ptp(t) == 0 or np.ptp(p) == 0 else float(spearmanr(t, p).statistic)
    return {"mae": float(np.mean(np.abs(t - p))), "spearman": rho}


def per_seed_scores(oof: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """Pooled scores per method, seed and target, over all samples or one level."""
    data = oof if subset == "all" else oof[oof["level"] == subset]
    rows = []
    for (method, seed), g in data.groupby(["method", "seed"], sort=False):
        for t in TARGETS:
            rows.append(
                {"subset": subset, "method": method, "seed": seed, "target": t, **score(g, t)}
            )
    return pd.DataFrame(rows)


def per_fold_scores(oof: pd.DataFrame, target: str = "score_mean") -> pd.DataFrame:
    """Per test player: MAE and Spearman for one target, mean over seeds."""
    rows = []
    for (method, seed, player), g in oof.groupby(["method", "seed", "test_player"], sort=False):
        rows.append({"method": method, "seed": seed, "test_player": player, **score(g, target)})
    df = pd.DataFrame(rows)
    return (
        df.groupby(["test_player", "method"], sort=False)[["mae", "spearman"]].mean().reset_index()
    )


def beats_level_baseline(scores: pd.DataFrame, method: str) -> dict[str, bool]:
    """Pre-registered rule: pooled MAE below the level-only baseline in every seed."""
    level = scores[scores["method"] == "level_mean"].set_index("target")["mae"]
    m = scores[scores["method"] == method]
    return {t: bool((g["mae"] < level[t]).all()) for t, g in m.groupby("target", sort=False)}
