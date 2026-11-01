"""Static review figures. Colors and marks follow the project's chart rules:
fixed categorical order, thin 2px lines, a legend plus a line style per series,
recessive grid and axes, one y-axis per panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from racketimu import io  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated, light mode)
STYLES = ["-", "--", ":"]  # second encoding, so identity is never color alone
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
PAD_FILL = "#f0efec"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))  # peak gyro row


def pick_examples(windows: pd.DataFrame, per_class: int, seed: int) -> dict[str, list[int]]:
    """Per class, sample ids from different players, mixing novices and pros."""
    rng = np.random.default_rng(seed)
    meta = windows.groupby("sample_id")[["player", "level", "stroke"]].first().reset_index()
    picks = {}
    for stroke in io.STROKES:
        rows = meta[meta["stroke"] == stroke].sample(frac=1, random_state=rng)
        by_level = [rows[rows["level"] == lv] for lv in ("novice", "pro")]
        chosen, players = [], set()
        for k in range(len(rows)):
            pool = by_level[k % 2] if not by_level[k % 2].empty else rows
            pool = pool[~pool["player"].isin(players)]
            if pool.empty or len(chosen) == per_class:
                continue
            chosen.append(int(pool["sample_id"].iloc[0]))
            players.add(pool["player"].iloc[0])
        picks[stroke] = chosen
    return picks


def plot_examples(
    windows: pd.DataFrame, picks: dict[str, list[int]], channels: list[str], unit: str, out: Path
) -> None:
    """Grid: one row per class, one column per example; real rows drawn, padding shaded."""
    n_cols = max(len(v) for v in picks.values())
    fig, axes = plt.subplots(
        len(picks), n_cols, figsize=(3.4 * n_cols, 2.4 * len(picks)), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    for r, (stroke, ids) in enumerate(picks.items()):
        for c in range(n_cols):
            ax = axes[r, c]
            _style_axes(ax)
            if c >= len(ids):
                ax.set_visible(False)
                continue
            w = windows[windows["sample_id"] == ids[c]]
            real = w[w["mask"]]
            for lo, hi in _padding_spans(w):
                ax.axvspan(lo - 0.5, hi + 0.5, color=PAD_FILL, linewidth=0)
            for k, ch in enumerate(channels):
                ax.plot(real["pos"], real[ch], color=SERIES[k], linestyle=STYLES[k],
                        linewidth=1.6, label=ch)  # fmt: skip
            meta = w.iloc[0]
            ax.set_title(f"{stroke.title()} · id {meta.sample_id} · {meta.player} ({meta.level})",
                         fontsize=8.5, color=INK, loc="left")  # fmt: skip
            if c == 0:
                ax.set_ylabel(unit, fontsize=8, color=INK_2)
            if r == len(picks) - 1:
                ax.set_xlabel("rows from peak gyro row", fontsize=8, color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(channels), fontsize=8, frameon=False)
    fig.suptitle(f"Example strokes per class: {', '.join(channels)} (shaded = padding)",
                 x=0.01, ha="left", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _padding_spans(w: pd.DataFrame) -> list[tuple[int, int]]:
    pad = w.loc[~w["mask"], "pos"].to_numpy()
    if pad.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(pad) != 1)
    starts = np.r_[pad[0], pad[breaks + 1]]
    ends = np.r_[pad[breaks], pad[-1]]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def plot_class_profiles(windows: pd.DataFrame, out: Path) -> None:
    """Median and interquartile band of gyro and acc magnitude per class (real rows only)."""
    w = windows[windows["mask"]].copy()
    w["gyro_mag"] = np.linalg.norm(w[io.GYRO].to_numpy(), axis=1)
    w["acc_mag"] = np.linalg.norm(w[io.ACC].to_numpy(), axis=1)
    panels = [("gyro_mag", "|gyro| (rad/s)"), ("acc_mag", "|acc| (m/s²)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (col, label) in zip(axes, panels, strict=True):
        _style_axes(ax)
        for k, stroke in enumerate(io.STROKES):
            g = w[w["stroke"] == stroke].groupby("pos")[col]
            q = g.quantile([0.25, 0.5, 0.75]).unstack()
            n = w.loc[w["stroke"] == stroke, "sample_id"].nunique()
            ax.fill_between(q.index, q[0.25], q[0.75], color=SERIES[k], alpha=0.15, linewidth=0)
            ax.plot(q.index, q[0.5], color=SERIES[k], linestyle=STYLES[k], linewidth=2,
                    label=f"{stroke.title()} (n={n})")  # fmt: skip
        ax.set_ylabel(label, fontsize=8, color=INK_2)
        ax.set_xlabel("rows from peak gyro row", fontsize=8, color=INK_2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Per-class median and interquartile range, real rows only",
                 x=0.01, ha="left", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_coverage(windows: pd.DataFrame, out: Path) -> None:
    """Share of strokes with a real (not padded) row at each window position, per class."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    for k, stroke in enumerate(io.STROKES):
        share = windows[windows["stroke"] == stroke].groupby("pos")["mask"].mean()
        ax.plot(share.index, share * 100, color=SERIES[k], linestyle=STYLES[k], linewidth=2,
                label=stroke.title())  # fmt: skip
    ax.set_ylim(0, 102)
    ax.set_ylabel("strokes with a real row (%)", fontsize=8, color=INK_2)
    ax.set_xlabel("rows from peak gyro row", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Window coverage per class", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
