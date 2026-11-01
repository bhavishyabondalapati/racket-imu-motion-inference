"""Fold-local scaling and the PyTorch training loop with early stopping (D3)."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from racketimu.seed import get_device, set_seed


def fit_scaler(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over the real rows of the training set only."""
    rows = x[mask]
    std = rows.std(axis=0)
    return rows.mean(axis=0), np.where(std > 0, std, 1.0)


def apply_scaler(x: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize, then put padded rows back to exactly zero."""
    out = ((x - mean) / std).astype(np.float32)
    out[~mask] = 0.0
    return out


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """'Balanced' weights: n_samples / (n_classes * count), as in scikit-learn."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)


def _tensors(x, mask, y=None, y_dtype=torch.long):
    out = [torch.from_numpy(x), torch.from_numpy(mask)]
    return out + ([torch.from_numpy(y).to(y_dtype)] if y is not None else [])


def _epoch_loss(model, loss_fn, x, m, y) -> float:
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(x, m), y))


def train_model(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    make_loss: Callable[[torch.device], nn.Module],
    cfg: dict,
    seed: int,
    y_dtype: torch.dtype = torch.long,
) -> tuple[nn.Module, dict]:
    """Train with Adam and mini-batches; keep the best epoch by validation loss.

    Stops when the validation loss has not improved for `early_stopping_patience` epochs.
    """
    set_seed(seed)
    device = get_device(cfg["device"])
    model.to(device)
    x_tr, m_tr, y_tr = (t.to(device) for t in _tensors(*train, y_dtype=y_dtype))
    x_va, m_va, y_va = (t.to(device) for t in _tensors(*val, y_dtype=y_dtype))
    loss_fn = make_loss(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    gen = torch.Generator().manual_seed(seed)
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for idx in torch.randperm(len(y_tr), generator=gen).split(cfg["batch_size"]):
            opt.zero_grad()
            loss_fn(model(x_tr[idx], m_tr[idx]), y_tr[idx]).backward()
            opt.step()
        val_loss = _epoch_loss(model, loss_fn, x_va, m_va, y_va)
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= cfg["early_stopping_patience"]:
            break
    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
        "epochs_run": epoch + 1,
    }


def train_classifier(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[nn.Module, dict]:
    """Class-weighted cross-entropy (D3)."""

    def make_loss(device):
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    return train_model(model, train, val, make_loss, cfg, seed)


def train_regressor(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: dict,
    seed: int,
) -> tuple[nn.Module, dict]:
    """Mean squared error on (standardized) continuous targets [N, K] (D4)."""
    return train_model(model, train, val, lambda device: nn.MSELoss(), cfg, seed, torch.float32)


def predict_logits(model: nn.Module, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt, mt = (t.to(device) for t in _tensors(x, mask))
        return model(xt, mt).cpu().numpy()
