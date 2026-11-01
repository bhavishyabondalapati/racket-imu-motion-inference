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
                ax.plot(
                    real["pos"],
                    real[ch],
                    color=SERIES[k],
                    linestyle=STYLES[k],
                    linewidth=1.6,
                    label=ch,
                )
            meta = w.iloc[0]
            ax.set_title(
                f"{stroke.title()} · id {meta.sample_id} · {meta.player} ({meta.level})",
                fontsize=8.5,
                color=INK,
                loc="left",
            )
            if c == 0:
                ax.set_ylabel(unit, fontsize=8, color=INK_2)
            if r == len(picks) - 1:
                ax.set_xlabel("rows from peak gyro row", fontsize=8, color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(channels), fontsize=8, frameon=False)
    fig.suptitle(
        f"Example strokes per class: {', '.join(channels)} (shaded = padding)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
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
            ax.plot(
                q.index,
                q[0.5],
                color=SERIES[k],
                linestyle=STYLES[k],
                linewidth=2,
                label=f"{stroke.title()} (n={n})",
            )
        ax.set_ylabel(label, fontsize=8, color=INK_2)
        ax.set_xlabel("rows from peak gyro row", fontsize=8, color=INK_2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Per-class median and interquartile range, real rows only",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
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
        ax.plot(
            share.index,
            share * 100,
            color=SERIES[k],
            linestyle=STYLES[k],
            linewidth=2,
            label=stroke.title(),
        )
    ax.set_ylim(0, 102)
    ax.set_ylabel("strokes with a real row (%)", fontsize=8, color=INK_2)
    ax.set_xlabel("rows from peak gyro row", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Window coverage per class", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def plot_confusions(matrices: dict[str, np.ndarray], classes: list[str], out: Path) -> None:
    """Row-normalized confusion matrices side by side; cells show count and row share."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", *BLUE_RAMP])
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.6 * len(matrices), 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, counts) in zip(np.atleast_1d(axes), matrices.items(), strict=True):
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ink = "#ffffff" if share[i, j] > 0.55 else INK
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]}\n{share[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=ink,
                )
        ax.set_xticks(range(len(classes)), [c.title() for c in classes], fontsize=8, color=INK_2)
        ax.set_yticks(range(len(classes)), [c.title() for c in classes], fontsize=8, color=INK_2)
        ax.set_xlabel("predicted", fontsize=8, color=INK_2)
        ax.set_ylabel("true", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Pooled out-of-fold confusion matrices (counts summed over seeds; % of true class)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(tables: dict[str, dict[str, pd.DataFrame]], out: Path) -> None:
    """Reliability diagrams: accuracy vs confidence per bin, before and after temperature T."""
    fig, axes = plt.subplots(1, len(tables), figsize=(3.6 * len(tables), 3.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, pair) in zip(np.atleast_1d(axes), tables.items(), strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (2, 2)), label="perfect")
        for k, (label, t) in enumerate(pair.items()):
            ax.plot(
                t["confidence"],
                t["accuracy"],
                color=SERIES[k],
                linestyle=STYLES[k],
                linewidth=2,
                marker="o",
                markersize=4,
                label=label,
            )
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("confidence (top class)", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("accuracy", fontsize=8, color=INK_2)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(
        "Reliability (pooled out-of-fold, all seeds)", x=0.01, ha="left", fontsize=10, color=INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_f1_vs_stride(summary: pd.DataFrame, cfg: dict, deep: str, out: Path) -> None:
    """Macro-F1 against row stride (mean ± std over seeds), RF and the deep model."""
    rate = 1.0 / np.array(cfg["ablation"]["row_strides"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.lines[-1].remove()  # no peak marker on this chart
    for k, (model, label) in enumerate([("rf", "Random forest"), (deep, deep.upper())]):
        s = summary[
            (summary["model"] == model) & summary["condition"].str.match(r"^(full|stride_\d+)$")
        ]
        stride = s["condition"].map(lambda c: 1 if c == "full" else int(c.split("_")[1]))
        order = np.argsort(stride.to_numpy())
        ax.errorbar(stride.to_numpy()[order], s["f1_mean"].to_numpy()[order],
                    yerr=s["f1_std"].to_numpy()[order], color=SERIES[k], linestyle=STYLES[k],
                    linewidth=2, marker="o", markersize=6, capsize=3, label=label)  # fmt: skip
    strides = cfg["ablation"]["row_strides"]
    ax.set_xticks(strides, [f"every {s}{'st' if s == 1 else 'th'} row\n(~{8 * r:.0f} rows/s)"
                            for s, r in zip(strides, rate, strict=True)])  # fmt: skip
    ax.set_xlim(0.5, max(strides) + 0.5)
    ax.set_ylabel("macro-F1 (pooled LOPO)", fontsize=8, color=INK_2)
    ax.set_title("Stroke-type macro-F1 vs row stride", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ablation_deltas(rf_summary: pd.DataFrame, out: Path) -> None:
    """Horizontal bars: change in RF macro-F1 from full data, per condition (± std over seeds)."""
    s = rf_summary[rf_summary["condition"] != "full"].sort_values("delta_mean")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    y = np.arange(len(s))
    ax.barh(y, s["delta_mean"], xerr=s["delta_std"], color=SERIES[0], height=0.6,
            error_kw={"ecolor": MUTED, "capsize": 2, "linewidth": 1})  # fmt: skip
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, s["label"], fontsize=8, color=INK_2)
    for yi, v in zip(y, s["delta_mean"], strict=True):
        ax.text(v + (0.005 if v >= 0 else -0.005), yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)  # fmt: skip
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.set_xlabel("change in macro-F1 from full data (random forest)", fontsize=8, color=INK_2)
    ax.set_title("Which reductions hurt most", fontsize=10, color=INK, loc="left")
    lo = min(s["delta_mean"].min() - s["delta_std"].max() - 0.05, -0.05)
    ax.set_xlim(lo, max(0.05, s["delta_mean"].max() + 0.05))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
