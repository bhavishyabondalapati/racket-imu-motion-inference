"""D6 sparsity ablation: how much sensor data does stroke-type classification need?

Every condition reuses the D3 protocol unchanged (same 15 LOPO folds, same inner
validation players, same models and hyperparameters, CPU, 3 seeds). Only the input
changes: which rows (stride), which channels, how long a window, or whether each
sample's channel means are removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from racketimu import io
from racketimu.classify import CLASSES, N_CLASSES, Inputs, make_folds, run_deep, run_forest
from racketimu.dataset import to_arrays
from racketimu.evaluate import classification_metrics
from racketimu.features import handcrafted_features, motion_only_features


@dataclass
class Condition:
    name: str
    group: str  # "rate", "sensors", "window", "motion"
    channels: str  # key into CHANNEL_SETS
    stride: int = 1
    window_fraction: float = 1.0
    motion_only: bool = False
    deep: bool = False  # also run the best D3 deep model
    label: str = ""
    extra: dict = field(default_factory=dict)


def channel_sets(cfg: dict) -> dict[str, list[str]]:
    primary, heading = cfg["channels"]["primary"], cfg["channels"]["heading"]
    return {
        "acc": io.ACC,
        "gyro": io.GYRO,
        "acc_gyro": io.ACC + io.GYRO,
        "primary": primary,
        "primary_heading": primary + heading,
    }


def conditions(cfg: dict) -> list[Condition]:
    """The D6 grid. Full data = primary channels, stride 1, 100% window."""
    strides = cfg["ablation"]["row_strides"]
    fractions = cfg["ablation"]["window_fractions"]
    out = [
        Condition(
            "full", "reference", "primary", deep=True, label="Full data (primary, stride 1, 100%)"
        ),
        *[
            Condition(
                f"stride_{k}",
                "rate",
                "primary",
                stride=k,
                deep=(k == max(strides)),
                label=f"Every {k}th row",
            )
            for k in strides
            if k != 1
        ],
        Condition("acc_only", "sensors", "acc", label="Accelerometer only"),
        Condition("gyro_only", "sensors", "gyro", label="Gyroscope only"),
        Condition("acc_gyro", "sensors", "acc_gyro", label="Accelerometer + gyroscope"),
        Condition(
            "primary_heading", "sensors", "primary_heading", label="Primary + heading (yaw, mag)"
        ),
        *[
            Condition(
                f"window_{int(f * 100)}",
                "window",
                "primary",
                window_fraction=f,
                label=f"Window {int(f * 100)}%",
            )
            for f in fractions
            if f != 1.0
        ],
        Condition(
            "motion_only",
            "motion",
            "primary",
            motion_only=True,
            deep=True,
            label="Motion-only (each sample's channel means removed)",
        ),
    ]
    return out


def select_rows(windows: pd.DataFrame, cond: Condition, cfg: dict) -> pd.DataFrame:
    """Apply the window length and stride to the D2 windows.

    Window: keep a shorter window with the peak at the same relative position
    (48 rows with the peak at 16 -> 36 rows with the peak at 12 -> 24 rows at 8).
    Stride: keep rows whose position from the peak is a multiple of k, so the peak row
    is always kept and the window spans the same time with k times fewer rows.
    """
    n_rows, peak = cfg["window"]["n_rows"], cfg["window"]["peak_position"]
    before = int(round(peak * cond.window_fraction))
    length = int(round(n_rows * cond.window_fraction))
    keep = (windows["pos"] >= -before) & (windows["pos"] < length - before)
    keep &= windows["pos"] % cond.stride == 0
    return windows[keep]


def condition_inputs(windows: pd.DataFrame, cond: Condition, cfg: dict) -> Inputs:
    channels = channel_sets(cfg)[cond.channels]
    x, mask, meta = to_arrays(select_rows(windows, cond, cfg), channels)
    if cond.motion_only:
        feats = motion_only_features(x, mask, channels)[0]
        means = np.nanmean(np.where(mask[..., None], x, np.nan), axis=1, keepdims=True)
        x = np.where(mask[..., None], x - means, 0.0).astype(np.float32)
    else:
        feats = handcrafted_features(x, mask, channels)[0]
    y = meta["stroke"].map({s: i for i, s in enumerate(CLASSES)}).to_numpy()
    return Inputs(meta=meta, y=y, x_primary=x, mask=mask, features={"rf": feats})


def run_condition(windows, cond, cfg, deep_model, log=print) -> pd.DataFrame:
    """Out-of-fold predictions for one condition: RF always, the deep model if asked."""
    inp = condition_inputs(windows, cond, cfg)
    rows = []
    for fold in make_folds(inp.meta, cfg):
        for seed in cfg["seeds"]:
            runs = [("rf", run_forest("rf", inp, fold, cfg, seed)[0])]
            if cond.deep:
                runs.append((deep_model, run_deep(deep_model, inp, fold, cfg, seed)[0]))
            for model, probs in runs:
                rows.append(
                    pd.DataFrame(
                        {
                            "condition": cond.name,
                            "model": model,
                            "seed": seed,
                            "test_player": fold.test_player,
                            "sample_id": inp.meta["sample_id"].to_numpy()[fold.test],
                            "y_true": inp.y[fold.test],
                            "y_pred": probs.argmax(axis=1),
                        }
                    )
                )
    log(
        f"  {cond.name}: {inp.x_primary.shape[1]} rows x {inp.x_primary.shape[2]} channels, "
        f"{inp.features['rf'].shape[1]} RF features"
    )
    return pd.concat(rows, ignore_index=True)


def summarize(oof: pd.DataFrame, conds: list[Condition]) -> pd.DataFrame:
    """Macro-F1 and accuracy per condition and model: mean ± std over seeds, and the
    difference from the full-data result of the same model."""
    per_seed = []
    for (cond, model, seed), g in oof.groupby(["condition", "model", "seed"]):
        m = classification_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), N_CLASSES)
        per_seed.append(
            {
                "condition": cond,
                "model": model,
                "seed": seed,
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            }
        )
    per_seed = pd.DataFrame(per_seed)
    full = per_seed[per_seed["condition"] == "full"].set_index(["model", "seed"])["macro_f1"]
    per_seed["delta"] = (
        per_seed["macro_f1"]
        - full.reindex(pd.MultiIndex.from_frame(per_seed[["model", "seed"]])).to_numpy()
    )
    g = per_seed.groupby(["condition", "model"])
    out = pd.DataFrame(
        {
            "f1_mean": g["macro_f1"].mean(),
            "f1_std": g["macro_f1"].std(),
            "acc_mean": g["accuracy"].mean(),
            "delta_mean": g["delta"].mean(),
            "delta_std": g["delta"].std(),
        }
    ).reset_index()
    meta = {c.name: (c.group, c.label) for c in conds}
    out["group"] = out["condition"].map(lambda c: meta[c][0])
    out["label"] = out["condition"].map(lambda c: meta[c][1])
    order = {c.name: i for i, c in enumerate(conds)}
    return out.sort_values(
        ["model", "condition"], key=lambda s: s.map(order) if s.name == "condition" else s
    )
