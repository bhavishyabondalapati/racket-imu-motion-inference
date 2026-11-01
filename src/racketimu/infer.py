"""Public inference interface (D8): one stroke in, stroke type out.

The deployed model is the D3 random forest (stroke type), retrained on every usable
sample by `make train-final`. A quality estimate (the D4 random-forest regressors) is
available only on explicit request, because it does not beat a level-only baseline.

CLI:
    python -m racketimu.infer --player P04 --out p04.json
    python -m racketimu.infer --csv data/raw/tabrizi2020/Raw/Sensory-Data/77.csv
    python -m racketimu.infer --player P04 --experimental-quality
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from racketimu import io
from racketimu.config import PROJECT_ROOT, load_config
from racketimu.dataset import add_angle_encodings, window_stroke
from racketimu.features import handcrafted_features

INPUT_COLUMNS = [*io.ACC, *io.GYRO, "roll_deg", "pitch_deg"]  # order expected in imu_window
QUALITY_TARGETS = [*io.CRITERIA, "score_mean"]
QUALITY_WARNING = "does not beat a level-only baseline; experimental"
MODEL_FILE = "stroke_models.joblib"


@dataclass
class InferenceResult:
    """What `infer` returns for one stroke."""

    stroke_type: str  # BASIC, PUSH or TOPSPIN
    probabilities: dict[str, float]  # per class, uncalibrated, sums to 1
    confidence: float  # probability of the predicted class
    confidence_is_calibrated: bool  # False: temperature scaling did not transfer (see D3)
    rows_used: int  # real rows in the model window after any rate reduction
    # Only with experimental_quality=True. Does not beat a level-only baseline (D4).
    quality: dict[str, float] | None = None  # predicted coach scores C1-C5, score_mean
    quality_note: str | None = None

    def to_dict(self) -> dict:
        """As a plain dict, leaving out the quality fields when they were not requested."""
        out = asdict(self)
        if self.quality is None:
            out.pop("quality")
            out.pop("quality_note")
        return out


@dataclass
class StrokeModels:
    classifier: object
    regressors: dict[str, object]
    channels: list[str]
    window: dict
    train_rate_hz: float
    quality_note: str


def _features(imu_window: np.ndarray, models: StrokeModels) -> tuple[np.ndarray, int]:
    stroke = add_angle_encodings(
        pd.DataFrame(imu_window, columns=INPUT_COLUMNS).assign(yaw_deg=0.0)
    )
    values, mask, _ = window_stroke(
        stroke, models.window["n_rows"], models.channels, models.window["peak_position"]
    )
    feats, _ = handcrafted_features(values[None], mask[None], models.channels)
    return feats, int(mask.sum())


def _resample(imu_window: np.ndarray, fs: float, train_rate: float) -> np.ndarray:
    """Keep every k-th row if fs is well above the training rate (k = round(fs / train_rate))."""
    stride = max(1, int(round(fs / train_rate)))
    if fs < 0.75 * train_rate:
        warnings.warn(
            f"fs={fs} Hz is below the training rate (~{train_rate:.1f} rows/s); "
            "see the D6 ablation for how accuracy changes with fewer rows.",
            stacklevel=3,
        )
    return imu_window[::stride]


def load_models(path: str | Path | None = None) -> StrokeModels:
    path = Path(path) if path else PROJECT_ROOT / "models" / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"No trained models at {path}. Run `make train-final` first.")
    return joblib.load(path)


def infer(
    imu_window: np.ndarray,
    fs: float,
    models: StrokeModels | None = None,
    experimental_quality: bool = False,
) -> InferenceResult:
    """Classify one forehand (Basic, Push or Topspin).

    Args:
        imu_window: one stroke's sensor rows, shape [T, 8], columns in this order:
            acc x/y/z (m/s²), gyro x/y/z (rad/s), roll (deg), pitch (deg), sensor frame,
            as logged by a BNO055 on the racket face. T >= 1.
        fs: rows per second of `imu_window`. The models were trained at about 8 rows/s;
            higher rates are reduced by keeping every k-th row.
        models: loaded models (defaults to `models/stroke_models.joblib`).
        experimental_quality: also return predicted coach scores. Off by default: the
            quality estimate does not beat a level-only baseline (D4); experimental.

    Returns:
        InferenceResult with the stroke type, per-class probabilities, the
        (uncalibrated) confidence, and, only if requested, predicted scores.

    Example:
        >>> import numpy as np
        >>> from racketimu.infer import infer
        >>> window = np.zeros((20, 8)); window[:, 2] = 9.81; window[10, 3] = 2.0
        >>> result = infer(window, fs=8.0)                        # doctest: +SKIP
        >>> result.stroke_type, round(result.confidence, 2)       # doctest: +SKIP
        ('BASIC', 0.61)
    """
    models = models or load_models()
    imu_window = np.asarray(imu_window, dtype=np.float64)
    if imu_window.ndim != 2 or imu_window.shape[1] != len(INPUT_COLUMNS) or len(imu_window) == 0:
        raise ValueError(
            f"imu_window must have shape [T, {len(INPUT_COLUMNS)}] with T >= 1, "
            f"columns {INPUT_COLUMNS}; got {imu_window.shape}"
        )
    if not np.isfinite(imu_window).all():
        raise ValueError("imu_window contains NaN or infinite values")
    feats, rows = _features(_resample(imu_window, fs, models.train_rate_hz), models)
    probs = models.classifier.predict_proba(feats)[0]
    classes = [str(c) for c in models.classifier.classes_]
    k = int(np.argmax(probs))
    result = InferenceResult(
        stroke_type=classes[k],
        probabilities={c: float(p) for c, p in zip(classes, probs, strict=True)},
        confidence=float(probs[k]),
        confidence_is_calibrated=False,
        rows_used=rows,
    )
    if experimental_quality:
        result.quality = {t: float(r.predict(feats)[0]) for t, r in models.regressors.items()}
        result.quality_note = f"{QUALITY_WARNING}. {models.quality_note}"
    return result


def stroke_to_window(stroke: pd.DataFrame) -> np.ndarray:
    """A loaded stroke (physical units, from racketimu.io) as an `imu_window` array."""
    return stroke[INPUT_COLUMNS].to_numpy(dtype=np.float64)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stroke inference and write JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--player", help="player code, e.g. P04 (all their used strokes)")
    group.add_argument("--csv", help="one raw stroke CSV in the dataset's format")
    parser.add_argument("--out", help="output JSON path (default: stdout)")
    parser.add_argument("--experimental-quality", action="store_true",
                        help=f"also predict coach scores ({QUALITY_WARNING})")  # fmt: skip
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    models = load_models()
    fs = models.train_rate_hz
    if args.csv:
        stroke = io.to_physical(io.read_raw_table(Path(args.csv)), cfg["dataset"]["units"])
        result = infer(stroke_to_window(stroke), fs, models, args.experimental_quality)
        items = [{"file": args.csv, **result.to_dict()}]
        note = (
            "The models were trained on every usable sample of the Tabrizi et al. (2020) "
            "dataset; if this file comes from that dataset, the prediction is in-sample."
        )
    else:
        ds = io.load_dataset(PROJECT_ROOT / cfg["dataset"]["root"], cfg["dataset"])
        inc = ds.included()
        rows = inc[inc["player"] == args.player.upper()]
        if rows.empty:
            print(f"ERROR: no used strokes for player {args.player}", file=sys.stderr)
            return 1
        items = []
        for r in rows.itertuples():
            result = infer(stroke_to_window(ds.strokes[r.sample_id]), fs, models,
                           args.experimental_quality)  # fmt: skip
            item = {"sample_id": int(r.sample_id), "true_stroke": r.stroke}
            if args.experimental_quality:
                item["true_score_mean"] = float(r.score_mean)
            items.append({**item, **result.to_dict()})
        note = (
            "These models were trained on all usable players, including this one, so these "
            "predictions are in-sample. Unseen-player performance is in the D3/D4 reports."
        )
    text = json.dumps({"note": note, "fs": fs, "strokes": items}, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {len(items)} predictions to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
