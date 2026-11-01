"""Metrics, calibration, and temperature scaling (D3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def probs_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-probabilities, so models that output probabilities can be temperature-scaled too."""
    return np.log(np.clip(probs, eps, 1.0))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 that minimizes the negative log-likelihood of softmax(logits / T)."""

    def nll(t: float) -> float:
        return -float(np.mean(log_softmax(logits / t, axis=1)[np.arange(len(y)), y]))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(logits / t, axis=1)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins (top-1 confidence)."""
    table = reliability_table(probs, y, n_bins)
    return float((table["count"] * (table["accuracy"] - table["confidence"]).abs()).sum() / len(y))


def reliability_table(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per confidence bin: sample count, mean confidence, accuracy (empty bins dropped)."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "conf": conf, "correct": pred == y})
    g = df.groupby("bin")
    return pd.DataFrame(
        {"count": g.size(), "confidence": g["conf"].mean(), "accuracy": g["correct"].mean()}
    ).reset_index()


def classification_metrics(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    """Accuracy, macro-F1 over all classes, and the confusion matrix (rows = true)."""
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def group_metrics(df: pd.DataFrame, by: str, classes: list[str]) -> pd.DataFrame:
    """Per group (e.g. test player): n, classes present, accuracy, macro-F1 over present classes.

    A player with one class has macro-F1 equal to that class's F1; classes the player
    does not have are left out of their average.
    """
    rows = []
    for key, g in df.groupby(by):
        present = sorted(g["y_true"].unique())
        rows.append({
            by: key, "n": len(g),
            "classes": ", ".join(classes[i] for i in present),
            "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
            "macro_f1_present": f1_score(g["y_true"], g["y_pred"], labels=present,
                                         average="macro", zero_division=0),
        })  # fmt: skip
    return pd.DataFrame(rows)
