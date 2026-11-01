"""D3: stroke-type classification under LOPO. Writes results/<run_id>/. Run with `make classify`.

Usage:
    python scripts/classify.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from racketimu import classify_report as rep
from racketimu.classify import CLASSES, run_all
from racketimu.config import PROJECT_ROOT, load_config
from racketimu.evaluate import reliability_table
from racketimu.plots import plot_confusions, plot_reliability

MAIN_MODELS = ["majority", "rf_primary", "cnn", "gru"]


def git_info() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout

    return {
        "commit": run("rev-parse", "HEAD").strip(),
        "dirty": bool(run("status", "--porcelain").strip()),
    }


def run_info(cfg: dict, seconds: float) -> dict:
    return {
        **git_info(),
        "duration_s": round(seconds, 1),
        "device": cfg["train"]["device"],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def write_report(out, oof, seed_metrics, checks, folds, cfg, info) -> None:
    lines = [
        f"# D3 stroke-type classification: {out.name}",
        "",
        f"Commit `{info['commit'][:10]}`{' (uncommitted changes)' if info['dirty'] else ''}; "
        f"device {info['device']}; {info['processor'] or info['machine']}; {info['duration_s']} s.",
        "",
        f"Protocol: LOPO over {len(folds)} test players (not used as test players: "
        f"{', '.join(cfg['lopo']['excluded_test_players'])}). In each fold, 1 novice and 1 pro "
        "training player form the inner validation set (early stopping, temperature scaling). "
        "Metrics pool all out-of-fold predictions; ± is the std over 3 seeds. Classes: "
        f"{', '.join(CLASSES)}.",
        "",
        "## Results",
        "",
        rep.md_table(rep.results_table(seed_metrics)),
        "",
        "## Shortcut checks",
        "",
        *rep.checks_markdown(checks),
        "",
        "## Per test player (macro-F1 over the classes that player has, mean over seeds)",
        "",
        rep.md_table(rep.per_player_table(oof, ["rf_primary", "rf_primary_heading", "cnn", "gru"])),
        "",
        "## Per level (pooled within level)",
        "",
        rep.md_table(rep.per_level_table(oof, MAIN_MODELS[1:]), index=False),
        "",
        "## Confusion matrices (pooled, summed over seeds)",
        "",
    ]
    for m in MAIN_MODELS[1:]:
        lines += [
            f"**{rep.MODEL_LABELS[m]}**",
            "",
            rep.confusion_markdown(rep.confusion_counts(oof, m)),
            "",
        ]
    lines += [
        "## Folds",
        "",
        rep.md_table(
            pd.DataFrame(
                [
                    {
                        "test player": f.test_player,
                        "validation players": ", ".join(f.val_players),
                        "fit samples": int(f.fit.sum()),
                        "val samples": int(f.val.sum()),
                        "test samples": int(f.test.sum()),
                    }
                    for f in folds
                ]
            ),
            index=False,
        ),
        "",
        "Figures: `confusion.png`, `reliability.png`. Data: `oof_predictions.csv.gz`, "
        "`metrics.json`, `config.yaml`, `run.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    windows_path = PROJECT_ROOT / cfg["paths"]["processed"] / "strokes.parquet"
    if not windows_path.exists():
        print(f"ERROR: {windows_path} not found. Run `make preprocess` first.", file=sys.stderr)
        return 1
    start = time.time()
    oof, folds = run_all(pd.read_parquet(windows_path), cfg)
    info = run_info(cfg, time.time() - start)

    out = PROJECT_ROOT / cfg["paths"]["results"] / f"d3_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    oof.to_csv(out / "oof_predictions.csv.gz", index=False)
    seed_metrics = rep.per_seed_metrics(oof, cfg["classify"]["ece_bins"])
    checks = rep.shortcut_checks(seed_metrics, oof, cfg)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (out / "run.json").write_text(json.dumps(info, indent=2))
    metrics = {
        "per_seed": seed_metrics.to_dict("records"),
        "checks": checks,
        "confusion": {m: rep.confusion_counts(oof, m).tolist() for m in MAIN_MODELS},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    plot_confusions(
        {rep.MODEL_LABELS[m].split(" (")[0]: rep.confusion_counts(oof, m) for m in MAIN_MODELS[1:]},
        CLASSES,
        out / "confusion.png",
    )
    bins = cfg["classify"]["ece_bins"]
    plot_reliability(
        {
            rep.MODEL_LABELS[m].split(" (")[0]: {
                "before T": reliability_table(
                    oof.loc[oof.model == m, rep.RAW].to_numpy(),
                    oof.loc[oof.model == m, "y_true"].to_numpy(),
                    bins,
                ),
                "after T": reliability_table(
                    oof.loc[oof.model == m, rep.CAL].to_numpy(),
                    oof.loc[oof.model == m, "y_true"].to_numpy(),
                    bins,
                ),
            }
            for m in MAIN_MODELS[1:]
        },
        out / "reliability.png",
    )
    write_report(out, oof, seed_metrics, checks, folds, cfg, info)
    print(rep.results_table(seed_metrics).to_string())
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}/ in {info['duration_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
