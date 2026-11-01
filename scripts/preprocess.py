"""Window every used stroke, save data/processed/strokes.parquet, and draw review figures (D2).

Usage:
    python scripts/preprocess.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from racketimu.config import PROJECT_ROOT, load_config
from racketimu.dataset import build_windows
from racketimu.io import ACC, EULER, GYRO, load_dataset
from racketimu.plots import pick_examples, plot_class_profiles, plot_coverage, plot_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    try:
        ds = load_dataset(PROJECT_ROOT / cfg["dataset"]["root"], cfg["dataset"])
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    windows = build_windows(ds, cfg["window"])
    out = PROJECT_ROOT / cfg["paths"]["processed"] / "strokes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out, index=False)

    per_sample = windows.groupby("sample_id")[["n_rows", "rows_cropped"]].first()
    n, cropped = len(per_sample), int((per_sample["rows_cropped"] > 0).sum())
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}: {n} strokes x {cfg['window']['n_rows']} rows")
    print(f"Real rows kept: {windows['mask'].sum()} of {per_sample['n_rows'].sum()} "
          f"({windows['mask'].sum() / per_sample['n_rows'].sum():.1%}); "
          f"{cropped} strokes lost rows to cropping")  # fmt: skip

    figs = PROJECT_ROOT / cfg["paths"]["figures"] / "d2"
    picks = pick_examples(windows, per_class=3, seed=cfg["seed"])
    plot_examples(windows, picks, GYRO, "rad/s", figs / "examples_gyro.png")
    plot_examples(windows, picks, ACC, "m/s²", figs / "examples_acc.png")
    plot_examples(windows, picks, EULER[:2], "degrees", figs / "examples_roll_pitch.png")
    plot_class_profiles(windows, figs / "class_profiles.png")
    plot_coverage(windows, figs / "window_coverage.png")
    print(f"Figures in {figs.relative_to(PROJECT_ROOT)}/ (examples: {picks})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
