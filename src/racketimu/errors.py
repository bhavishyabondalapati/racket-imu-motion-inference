"""D7 error analysis on the D3 out-of-fold predictions of one model.

Probabilities are averaged over the seeds for each sample, so the ranking of
confident and unconfident strokes does not depend on one seed. Every statement
built here is a count or a rate computed from the predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from racketimu.classify import CLASSES


def seed_averaged(oof: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per sample: mean uncalibrated probabilities, predicted class, confidence."""
    cols = [f"p_raw_{c}" for c in CLASSES]
    g = oof[oof["model"] == model].groupby("sample_id")
    df = g[cols].mean()
    df[["test_player", "level", "y_true"]] = g[["test_player", "level", "y_true"]].first()
    probs = df[cols].to_numpy()
    df["y_pred"] = probs.argmax(axis=1)
    df["confidence"] = probs.max(axis=1)
    df["true"] = df["y_true"].map(dict(enumerate(CLASSES)))
    df["pred"] = df["y_pred"].map(dict(enumerate(CLASSES)))
    df["correct"] = df["y_true"] == df["y_pred"]
    return df.reset_index()


def attach_window_info(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    info = windows.groupby("sample_id")[["n_rows", "peak_row", "rows_cropped"]].first()
    return df.join(info, on="sample_id")


def lowest_confidence(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nsmallest(n, "confidence")


def confident_errors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df[~df["correct"]].nlargest(n, "confidence")


def per_player(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per test player, with level and class mix."""
    g = df.groupby("test_player")
    out = pd.DataFrame({
        "level": g["level"].first(), "n": g.size(), "accuracy": g["correct"].mean().round(3),
        "classes": g["true"].apply(lambda s: ", ".join(sorted(s.unique()))),
        "errors": g["correct"].apply(lambda s: int((~s).sum())),
    })  # fmt: skip
    return out.sort_values("accuracy")


def error_rates(df: pd.DataFrame, by: pd.Series, name: str) -> pd.DataFrame:
    g = df.assign(group=by).groupby("group", observed=True)
    return pd.DataFrame(
        {"n": g.size(), "error rate": (1 - g["correct"].mean()).round(3)}
    ).rename_axis(name)


def confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    e = df[~df["correct"]]
    return (
        e.groupby(["true", "pred"])
        .size()
        .rename("errors")
        .reset_index()
        .sort_values("errors", ascending=False)
    )


def level_gap(players: pd.DataFrame) -> dict[str, float]:
    """Mean per-player accuracy by level (players weighted equally)."""
    return players.groupby("level")["accuracy"].mean().round(3).to_dict()


def pitch_degrees(windows: pd.DataFrame) -> pd.Series:
    return np.degrees(np.arctan2(windows["pitch_sin"], windows["pitch_cos"]))
