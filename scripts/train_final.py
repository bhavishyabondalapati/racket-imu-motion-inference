"""Train the deployable models on every usable sample (D8). Writes models/stroke_models.joblib.

Usage:
    python scripts/train_final.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from racketimu.config import PROJECT_ROOT, load_config
from racketimu.datacard import rows_per_whole_second
from racketimu.dataset import to_arrays
from racketimu.features import handcrafted_features
from racketimu.infer import MODEL_FILE, QUALITY_TARGETS, StrokeModels
from racketimu.io import load_dataset


def quality_note(cfg: dict) -> str:
    """Summarize the latest D4 verdict for the RF regressor, or say it is unknown."""
    runs = sorted((PROJECT_ROOT / cfg["paths"]["results"]).glob("d4_*"))
    if not runs:
        return "Quality accuracy on unseen players not yet evaluated (no D4 run found)."
    verdict = json.loads((runs[-1] / "metrics.json").read_text())["beats_level_baseline"]["rf"]
    beaten = [t for t, ok in verdict.items() if ok]
    if not beaten:
        return (
            f"On unseen players this estimate did not beat a level-only baseline on any score "
            f"(D4, {runs[-1].name}); treat it as mostly reflecting player level."
        )
    return (
        f"On unseen players this estimate beat a level-only baseline on {', '.join(beaten)} "
        f"(D4, {runs[-1].name}); see the D4 report for its error."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    path = PROJECT_ROOT / cfg["paths"]["processed"] / "strokes.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run `make preprocess` first.", file=sys.stderr)
        return 1
    channels = cfg["channels"]["primary"]
    x, mask, meta = to_arrays(pd.read_parquet(path), channels)
    feats = handcrafted_features(x, mask, channels)[0]
    rf = cfg["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=rf["n_estimators"],
        max_depth=rf["max_depth"],
        class_weight=cfg["classify"]["class_weight"],
        random_state=cfg["seed"],
        n_jobs=-1,
    ).fit(feats, meta["stroke"])
    regs = {
        t: RandomForestRegressor(
            n_estimators=rf["n_estimators"],
            max_depth=rf["max_depth"],
            random_state=cfg["seed"],
            n_jobs=-1,
        ).fit(feats, meta[t])
        for t in QUALITY_TARGETS
    }
    for m in [clf, *regs.values()]:
        m.set_params(n_jobs=1)  # single-stroke inference: threads only add overhead
    ds = load_dataset(PROJECT_ROOT / cfg["dataset"]["root"], cfg["dataset"])
    rate = float(np.mean(rows_per_whole_second(ds.strokes[i] for i in ds.included()["sample_id"])))
    models = StrokeModels(clf, regs, channels, dict(cfg["window"]), rate, quality_note(cfg))
    out = PROJECT_ROOT / "models" / MODEL_FILE
    out.parent.mkdir(exist_ok=True)
    joblib.dump(models, out, compress=3)
    print(
        f"Wrote {out.relative_to(PROJECT_ROOT)} ({out.stat().st_size / 1e6:.1f} MB): trained on "
        f"{len(meta)} strokes from {meta['player'].nunique()} players; rate {rate:.2f} rows/s."
    )
    print(f"Quality note: {models.quality_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
