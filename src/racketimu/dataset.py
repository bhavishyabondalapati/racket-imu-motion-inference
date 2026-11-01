"""Fixed-length stroke windows (D2) and the arrays models train on.

Each stroke file is one sample. It is padded or cropped to `n_rows` rows with the
row of peak gyro magnitude at `peak_position` (default n_rows // 2), and a mask
marks real rows.
Angles that wrap at ±180° (pitch, yaw) are also given as sin/cos so the inputs
stay continuous.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from racketimu import io

ANGLE_ENCODINGS = {"pitch_deg": "pitch", "yaw_deg": "yaw"}  # wrap at ±180° / 0-360°
CHANNELS = [
    *io.ACC, *io.GYRO, *io.EULER, "pitch_sin", "pitch_cos", "yaw_sin", "yaw_cos", *io.MAG,
]  # fmt: skip
META = ["sample_id", "player", "level", "stroke", *io.CRITERIA, "score_mean"]


def add_angle_encodings(stroke: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos columns for the angles that wrap (pitch, yaw)."""
    out = stroke.copy()
    for col, name in ANGLE_ENCODINGS.items():
        rad = np.deg2rad(out[col].to_numpy(float))
        out[f"{name}_sin"], out[f"{name}_cos"] = np.sin(rad), np.cos(rad)
    return out


def peak_row(stroke: pd.DataFrame) -> int:
    """Index of the first row with the largest gyro magnitude."""
    return int(np.argmax(np.linalg.norm(stroke[io.GYRO].to_numpy(float), axis=1)))


def window_bounds(
    n_stroke: int, peak: int, n_rows: int, peak_position: int | None = None
) -> tuple[int, int, int]:
    """Source rows [start, stop) to copy, and where they land in the window.

    The peak lands at `peak_position` (default n_rows // 2); rows outside the stroke
    are padding.
    """
    start = peak - (n_rows // 2 if peak_position is None else peak_position)
    src_start, src_stop = max(start, 0), min(start + n_rows, n_stroke)
    return src_start, src_stop, src_start - start


def window_stroke(
    stroke: pd.DataFrame,
    n_rows: int,
    channels: list[str] = CHANNELS,
    peak_position: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return (values [n_rows, channels], mask [n_rows], info) for one stroke.

    Padded rows are zero and have mask False. `info` records the raw peak row and how
    many real rows were cropped away.
    """
    if len(stroke) == 0:
        raise ValueError("Cannot window an empty stroke")
    peak = peak_row(stroke)
    src_start, src_stop, dst = window_bounds(len(stroke), peak, n_rows, peak_position)
    values = np.zeros((n_rows, len(channels)), dtype=np.float32)
    mask = np.zeros(n_rows, dtype=bool)
    n_copy = src_stop - src_start
    values[dst : dst + n_copy] = stroke[channels].to_numpy(np.float32)[src_start:src_stop]
    mask[dst : dst + n_copy] = True
    info = {"peak_row": peak, "rows_cropped": len(stroke) - n_copy}
    return values, mask, info


def build_windows(ds: io.Dataset, window_cfg: dict[str, Any]) -> pd.DataFrame:
    """Window every included stroke into a long table: one row per (sample, position)."""
    if window_cfg.get("center", "peak_gyro") != "peak_gyro":
        raise ValueError(f"Unsupported window center: {window_cfg['center']!r}")
    n_rows = int(window_cfg["n_rows"])
    peak_position = int(window_cfg.get("peak_position", n_rows // 2))
    parts = []
    for meta in ds.included()[[*META, "n_rows"]].itertuples(index=False):
        stroke = add_angle_encodings(ds.strokes[meta.sample_id])
        values, mask, info = window_stroke(stroke, n_rows, peak_position=peak_position)
        part = pd.DataFrame(values, columns=CHANNELS)
        part.insert(0, "pos", np.arange(n_rows) - peak_position)  # 0 = peak row
        part.insert(1, "mask", mask)
        for col in reversed([*META, "n_rows"]):
            part.insert(0, col, getattr(meta, col))
        part["peak_row"], part["rows_cropped"] = info["peak_row"], info["rows_cropped"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def to_arrays(
    windows: pd.DataFrame, channels: list[str]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reshape the long table into X [N, T, C], mask [N, T], and one metadata row per sample."""
    ordered = windows.sort_values(["sample_id", "pos"])
    n_samples = ordered["sample_id"].nunique()
    n_rows = len(ordered) // n_samples
    x = ordered[channels].to_numpy(np.float32).reshape(n_samples, n_rows, len(channels))
    mask = ordered["mask"].to_numpy(bool).reshape(n_samples, n_rows)
    meta = ordered.groupby("sample_id", sort=True)[[*META[1:], "n_rows"]].first().reset_index()
    return x, mask, meta


def lopo_test_players(meta: pd.DataFrame, excluded: list[str]) -> list[str]:
    """Players used as LOPO test folds: every player with samples, minus the excluded ones."""
    return sorted(p for p in meta["player"].unique() if p not in set(excluded))


def inner_validation_players(
    train_meta: pd.DataFrame, per_level: dict[str, int], seed: int
) -> list[str]:
    """Pick validation players from the training players only, per level.

    Players with all three stroke classes are preferred, so early stopping and
    temperature scaling see every class. The choice depends only on `seed`.
    """
    rng = np.random.default_rng(seed)
    classes = train_meta.groupby("player")["stroke"].nunique()
    chosen: list[str] = []
    for level, n in per_level.items():
        players = sorted(train_meta.loc[train_meta["level"] == level, "player"].unique())
        full = [p for p in players if classes[p] == len(io.STROKES)]
        pool = full if len(full) >= n else players
        chosen += sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    return chosen
