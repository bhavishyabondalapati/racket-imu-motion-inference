"""Summaries, checks, and the Markdown report for D3, computed from out-of-fold predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from racketimu.classify import CLASSES, N_CLASSES
from racketimu.datacard import md_table
from racketimu.evaluate import classification_metrics, ece, group_metrics

MODEL_LABELS = {
    "majority": "Majority class",
    "rf_mean_only": "Leak check: RF on per-sample channel means",
    "rf_motion_only": "Motion-only check: RF with each sample's channel means removed",
    "rf_length_only": "Length check: RF on real-row count",
    "rf_primary": "RF, handcrafted features (primary channels)",
    "rf_primary_heading": "Heading check: RF + yaw and magnetometer",
    "cnn": "1D CNN (primary channels)",
    "gru": "GRU (primary channels)",
}
RAW = [f"p_raw_{c}" for c in CLASSES]
CAL = [f"p_cal_{c}" for c in CLASSES]


def per_seed_metrics(oof: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Pooled out-of-fold metrics for every (model, seed)."""
    rows = []
    for (model, seed), g in oof.groupby(["model", "seed"], sort=False):
        y = g["y_true"].to_numpy()
        m = classification_metrics(y, g["y_pred"].to_numpy(), N_CLASSES)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "ece_raw": ece(g[RAW].to_numpy(), y, n_bins),
                "ece_cal": ece(g[CAL].to_numpy(), y, n_bins),
                "temperature_median": g.groupby("test_player")["temperature"].first().median(),
            }
        )
    return pd.DataFrame(rows)


def _pm(s: pd.Series, digits: int = 3) -> str:
    return (
        f"{s.mean():.{digits}f}" if len(s) == 1 else f"{s.mean():.{digits}f} ± {s.std():.{digits}f}"
    )


def results_table(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds (std across 3 seeds; majority has one deterministic run)."""
    g = seed_metrics.groupby("model", sort=False)
    out = pd.DataFrame(
        {
            "seeds": g.size(),
            "accuracy": g["accuracy"].apply(_pm),
            "macro-F1": g["macro_f1"].apply(_pm),
            "ECE (uncalibrated, primary)": g["ece_raw"].apply(_pm),
            "ECE after T (did not transfer)": g["ece_cal"].apply(_pm),
            "median T": g["temperature_median"].apply(lambda s: f"{s.median():.2f}"),
        }
    )
    out.index = [MODEL_LABELS[m] for m in out.index]
    return out.rename_axis("model")


def per_player_table(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per test player: macro-F1 over the classes that player has, mean over seeds."""
    parts = []
    for model in models:
        g = oof[oof["model"] == model]
        per = pd.concat(
            [group_metrics(s, "test_player", CLASSES).assign(seed=k) for k, s in g.groupby("seed")]
        )
        agg = per.groupby("test_player").agg(
            n=("n", "first"), classes=("classes", "first"), f1=("macro_f1_present", "mean")
        )
        parts.append(agg["f1"].rename(model))
        base = agg[["n", "classes"]]
    level = oof.groupby("test_player")["level"].first()
    table = pd.concat([base, level, *parts], axis=1).round(3)
    return table.rename(columns={m: MODEL_LABELS[m].split(" (")[0] for m in models})


def per_level_table(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        for level in ("novice", "pro"):
            g = oof[(oof["model"] == model) & (oof["level"] == level)]
            f1 = g.groupby("seed").apply(
                lambda s: classification_metrics(s["y_true"], s["y_pred"], N_CLASSES)["macro_f1"],
                include_groups=False,
            )
            acc = g.groupby("seed").apply(
                lambda s: (s["y_true"] == s["y_pred"]).mean(), include_groups=False
            )
            rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "level": level,
                    "n": len(g) // g["seed"].nunique(),
                    "accuracy": _pm(acc),
                    "macro-F1": _pm(f1),
                }
            )
    return pd.DataFrame(rows)


def confusion_counts(oof: pd.DataFrame, model: str) -> np.ndarray:
    """Pooled confusion matrix summed over seeds (rows = true class)."""
    g = oof[oof["model"] == model]
    return np.array(classification_metrics(g["y_true"], g["y_pred"], N_CLASSES)["confusion"])


def shortcut_checks(seed_metrics: pd.DataFrame, oof: pd.DataFrame, cfg: dict) -> dict:
    """Heading and leak checks against the thresholds fixed in the config before the run."""
    f1 = seed_metrics.pivot(index="seed", columns="model", values="macro_f1")
    majority = float(f1["majority"].dropna().iloc[0])
    gain = f1["rf_primary_heading"] - f1["rf_primary"]
    players = per_player_table(oof, ["rf_primary", "rf_primary_heading"])
    cols = players.columns[-2:]
    thr = cfg["classify"]["checks"]
    return {
        "heading_gain_mean": float(gain.mean()),
        "heading_gain_per_seed": gain.round(4).tolist(),
        "heading_players_improved": int((players[cols[1]] > players[cols[0]]).sum()),
        "heading_players_total": len(players),
        "heading_flagged": bool(gain.mean() >= thr["heading_gain_flag"]),
        "majority_macro_f1": majority,
        "mean_only_margin": float(f1["rf_mean_only"].mean() - majority),
        "mean_only_flagged": bool(f1["rf_mean_only"].mean() - majority >= thr["leak_margin_flag"]),
        "length_only_margin": float(f1["rf_length_only"].mean() - majority),
        "length_only_flagged": bool(
            f1["rf_length_only"].mean() - majority >= thr["leak_margin_flag"]
        ),
        "motion_only_macro_f1": float(f1["rf_motion_only"].mean()),
        "motion_only_vs_full_rf": float(f1["rf_motion_only"].mean() - f1["rf_primary"].mean()),
        "mean_only_vs_full_rf": float(f1["rf_mean_only"].mean() - f1["rf_primary"].mean()),
        "thresholds": thr,
    }


def confusion_markdown(counts: np.ndarray) -> str:
    share = counts / counts.sum(axis=1, keepdims=True)
    cells = [
        [f"{c} ({s:.0%})" for c, s in zip(row, srow, strict=True)]
        for row, srow in zip(counts, share, strict=True)
    ]
    df = pd.DataFrame(
        cells, index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES]
    )
    return md_table(df.rename_axis(""))


def checks_markdown(chk: dict) -> list[str]:
    thr = chk["thresholds"]
    flag = {True: "**FLAGGED**", False: "not flagged"}
    return [
        f"- **Heading check:** adding yaw and the magnetometer changes RF macro-F1 by "
        f"{chk['heading_gain_mean']:+.3f} on average (per seed: {chk['heading_gain_per_seed']}); "
        f"{chk['heading_players_improved']} of {chk['heading_players_total']} test players "
        f"improve. Rule fixed before the run: a gain of at least {thr['heading_gain_flag']} "
        f"counts as a likely shortcut. Result: {flag[chk['heading_flagged']]}.",
        f"- **Leak check (per-sample channel means only):** macro-F1 is "
        f"{chk['mean_only_margin']:+.3f} relative to the majority baseline "
        f"({chk['majority_macro_f1']:.3f}). Rule: a margin of at least {thr['leak_margin_flag']} "
        f"counts as well above. Result: {flag[chk['mean_only_flagged']]}.",
        f"- **Motion-only check (each sample's channel means removed; no threshold fixed in "
        f"advance):** macro-F1 {chk['motion_only_macro_f1']:.3f}, which is "
        f"{chk['motion_only_vs_full_rf']:+.3f} relative to the full handcrafted RF and "
        f"{chk['motion_only_macro_f1'] - chk['majority_macro_f1']:+.3f} relative to majority. "
        f"For comparison, the mean-only RF is {chk['mean_only_vs_full_rf']:+.3f} relative to "
        "the full RF.",
        f"- **Length check (real-row count only):** macro-F1 is "
        f"{chk['length_only_margin']:+.3f} relative to the majority baseline. Same rule. "
        f"Result: {flag[chk['length_only_flagged']]}.",
    ]
