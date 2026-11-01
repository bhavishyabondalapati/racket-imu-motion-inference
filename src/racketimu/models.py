"""Small sequence classifiers (D3). Both take x [B, T, C] and mask [B, T].

The mask is appended as an extra input channel so the model knows which rows are
padding, and the final pooling averages over real rows only.
"""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average h [B, T, H] over the time steps where mask [B, T] is True."""
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


class StrokeCNN(nn.Module):
    """Two 1D convolutions, masked mean pooling, and a linear head."""

    def __init__(self, n_channels: int, n_classes: int, channels: list[int], kernel_size: int,
                 dropout: float) -> None:  # fmt: skip
        super().__init__()
        layers, c_in = [], n_channels + 1
        for c_out in channels:
            layers += [nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, mask.unsqueeze(-1).to(x.dtype)], dim=-1).transpose(1, 2)
        h = self.conv(inp).transpose(1, 2)
        return self.head(masked_mean(h, mask))


class StrokeGRU(nn.Module):
    """One GRU layer over the window, masked mean pooling, and a linear head."""

    def __init__(self, n_channels: int, n_classes: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(n_channels + 1, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, mask.unsqueeze(-1).to(x.dtype)], dim=-1)
        h, _ = self.gru(inp)
        return self.head(masked_mean(h, mask))


def build_model(name: str, n_channels: int, n_classes: int, cfg: dict) -> nn.Module:
    if name == "cnn":
        return StrokeCNN(n_channels, n_classes, **cfg["cnn"])
    if name == "gru":
        return StrokeGRU(n_channels, n_classes, **cfg["gru"])
    raise ValueError(f"Unknown model: {name!r}")
