"""Handcrafted per-stroke features for the random-forest baselines (D3).

All statistics use real rows only (mask True). Every feature is computed from one
sample at a time, so nothing here can leak information between samples or folds.
"""

from __future__ import annotations

import numpy as np

from racketimu import io

GRAVITY = 9.81


def _masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of x [N, T, C] with padded rows set to NaN."""
    out = x.astype(np.float64)
    out[~mask] = np.nan
    return out


def _first_last(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last real value per sample for a [N, T] array."""
    first = mask.argmax(axis=1)
    last = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    rows = np.arange(len(values))
    return values[rows, first], values[rows, last]


def channel_stats(
    x: np.ndarray, mask: np.ndarray, channels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Mean, std, min and max of every channel over real rows."""
    xm = _masked(x, mask)
    stats = {"mean": np.nanmean, "std": np.nanstd, "min": np.nanmin, "max": np.nanmax}
    feats = [fn(xm, axis=1) for fn in stats.values()]
    names = [f"{c}_{s}" for s in stats for c in channels]
    return np.concatenate(feats, axis=1), names


def motion_features(
    x: np.ndarray, mask: np.ndarray, channels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Peak angular velocity, window energy, and orientation change.

    Each group is computed only if its channels are present (gyro x/y/z, acc x/y/z,
    roll, pitch sin/cos), so channel subsets in the D6 ablation work too.
    """
    idx = {c: i for i, c in enumerate(channels)}
    xm = _masked(x, mask)
    feats: dict[str, np.ndarray] = {}
    if all(c in idx for c in io.GYRO):
        gyro = np.linalg.norm(xm[:, :, [idx[c] for c in io.GYRO]], axis=2)
        feats["gyro_mag_peak"] = np.nanmax(gyro, axis=1)
        feats["gyro_mag_mean"] = np.nanmean(gyro, axis=1)
        feats["gyro_energy"] = np.nanmean(gyro**2, axis=1)
    if all(c in idx for c in io.ACC):
        acc = np.linalg.norm(xm[:, :, [idx[c] for c in io.ACC]], axis=2)
        feats["acc_mag_mean"] = np.nanmean(acc, axis=1)
        feats["acc_mag_std"] = np.nanstd(acc, axis=1)
        feats["acc_mag_peak"] = np.nanmax(acc, axis=1)
        feats["acc_dynamic_energy"] = np.nanmean((acc - GRAVITY) ** 2, axis=1)
    has_roll, has_pitch = "roll_deg" in idx, "pitch_sin" in idx and "pitch_cos" in idx
    if has_roll:
        roll_first, roll_last = _first_last(x[:, :, idx["roll_deg"]], mask)
        feats["roll_change"] = roll_last - roll_first
    if has_pitch:
        pitch = np.arctan2(x[:, :, idx["pitch_sin"]], x[:, :, idx["pitch_cos"]])
        pitch_first, pitch_last = _first_last(pitch, mask)
        feats["pitch_change"] = np.degrees(np.angle(np.exp(1j * (pitch_last - pitch_first))))
    if has_roll:
        feats["roll_path"] = np.nansum(np.abs(np.diff(xm[:, :, idx["roll_deg"]], axis=1)), axis=1)
    if has_pitch:
        steps = np.angle(np.exp(1j * np.diff(np.where(mask, pitch, np.nan), axis=1)))
        feats["pitch_path"] = np.degrees(np.nansum(np.abs(steps), axis=1))
    if not feats:
        return np.empty((len(x), 0)), []
    return np.column_stack(list(feats.values())), list(feats)


def handcrafted_features(
    x: np.ndarray, mask: np.ndarray, channels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Channel statistics for every channel, plus motion features from the primary channels."""
    stats, stat_names = channel_stats(x, mask, channels)
    motion, motion_names = motion_features(x, mask, channels)
    return np.concatenate([stats, motion], axis=1), stat_names + motion_names


def mean_features(
    x: np.ndarray, mask: np.ndarray, channels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Per-sample mean of each channel only: no information about the motion's shape."""
    return np.nanmean(_masked(x, mask), axis=1), [f"{c}_mean" for c in channels]


def length_features(mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Number of real rows in the window only."""
    return mask.sum(axis=1, keepdims=True).astype(np.float64), ["real_rows"]


def motion_only_features(
    x: np.ndarray, mask: np.ndarray, channels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Features that ignore each sample's average level ("motion-only" check).

    Each channel has its per-sample mean over real rows subtracted first, so static
    racket orientation is removed. Then: std, min and max of every centered channel;
    magnitude features of the centered gyro and acc vectors; and roll/pitch change and
    path, which are differences and so already ignore the mean. Adding a constant to a
    channel of one sample leaves every feature unchanged.
    """
    xm = _masked(x, mask)
    centered = x - np.nanmean(xm, axis=1, keepdims=True)
    centered[~mask] = 0.0
    stats, names = channel_stats(centered, mask, channels)
    keep = [i for i, n in enumerate(names) if not n.endswith("_mean")]
    motion, motion_names = motion_features(centered, mask, channels)
    from_original, orig_names = motion_features(x, mask, channels)
    angle_cols = [i for i, n in enumerate(motion_names) if n.startswith(("roll_", "pitch_"))]
    for i in angle_cols:  # angle changes are taken from the uncentered angles
        motion[:, i] = from_original[:, orig_names.index(motion_names[i])]
    motion_names = [
        f"{n}_centered" if i not in angle_cols else n for i, n in enumerate(motion_names)
    ]
    return np.concatenate([stats[:, keep], motion], axis=1), [names[i] for i in keep] + motion_names
