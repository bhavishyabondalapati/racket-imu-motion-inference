"""D4: stroke-quality estimation (C1-C5 and their mean) under LOPO. Writes results/<run_id>/.

Usage:
    python scripts/quality.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from racketimu import quality as Q
from racketimu.config import PROJECT_ROOT, load_config
from racketimu.datacard import md_table
from racketimu.plots import plot_quality_scatter
from racketimu.runinfo import git_state, new_run_dir, provenance_line, write_run_files

LABELS = {
    "global_mean": "Baseline: global training mean",
    "class_mean": "Baseline: per-class training mean",
    "level_mean": "Baseline: level-only (training mean of test player's level)",
    "rf": "RF regressor (handcrafted primary features)",
}


def metric_table(scores: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    """Rows = method, columns = target; mean ± std over seeds (baselines are deterministic)."""
    out = {}
    for method in methods:
        s = scores[scores["method"] == method].groupby("target", sort=False)[metric]
        cells = {}
        for t, g in s:
            if g.isna().all():
                cells[t] = "n/a"
            elif len(g) == 1:
                cells[t] = f"{g.mean():.2f}" if metric == "mae" else f"{g.mean():.3f}"
            else:
                fmt = ".2f" if metric == "mae" else ".3f"
                cells[t] = f"{g.mean():{fmt}} ± {g.std():{fmt}}"
        out[LABELS.get(method, method.upper())] = cells
    return pd.DataFrame(out).T[Q.TARGETS].rename_axis("method")


def verdict_lines(scores: pd.DataFrame, models: list[str]) -> list[str]:
    lines = []
    for m in models:
        beat = Q.beats_level_baseline(scores, m)
        yes = [t for t, b in beat.items() if b]
        lines.append(
            f"- **{LABELS.get(m, m.upper())}:** beats the level-only baseline on "
            f"{len(yes)} of {len(beat)} targets" + (f" ({', '.join(yes)})." if yes else ".")
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    path = PROJECT_ROOT / cfg["paths"]["processed"] / "strokes.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run `make preprocess` first.", file=sys.stderr)
        return 1
    git, start = git_state(), time.time()
    oof = Q.run_all(pd.read_parquet(path), cfg)
    out = new_run_dir(cfg, "d4")
    info = write_run_files(out, cfg, git, time.time() - start)
    oof.to_csv(out / "oof_predictions.csv.gz", index=False)
    deep = cfg["ablation"]["deep_model"]
    methods = [*Q.BASELINES, "rf", deep]
    scores = {s: Q.per_seed_scores(oof, s) for s in ("all", "novice", "pro")}
    folds = Q.per_fold_scores(oof)
    verdict = {m: Q.beats_level_baseline(scores["all"], m) for m in ("rf", deep)}
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "scores": pd.concat(scores.values()).to_dict("records"),
                "beats_level_baseline": verdict,
                "per_fold_score_mean": folds.to_dict("records"),
            },
            indent=2,
            default=float,
        )
    )
    plot_quality_scatter(oof, "rf", out / "quality_scatter.png")

    lines = [
        f"# D4 stroke-quality estimation: {out.name}",
        "",
        provenance_line(info),
        "",
        "Same 15 LOPO folds and inner validation players as D3; 3 seeds for the models. "
        "Baselines use all training players; the RF regressor uses the fit set; the "
        f"{deep.upper()} uses the fit set with the validation players for early stopping. "
        "Scores are the coach averages (0-100) of criteria C1-C5 and their mean. No coach-"
        "agreement ceiling exists (individual coach scores are not published).",
        "",
        "## Does the model judge stroke quality beyond level?",
        "",
        "Rule fixed before the run: a model beats the level-only baseline on a target only if "
        "its pooled MAE is lower in every seed.",
        "",
        *verdict_lines(scores["all"], ["rf", deep]),
        "",
        "## All test samples, pooled",
        "",
        "**MAE (lower is better)**",
        "",
        md_table(metric_table(scores["all"], "mae", methods)),
        "",
        "**Spearman correlation (higher is better)**",
        "",
        md_table(metric_table(scores["all"], "spearman", methods)),
        "",
        "## Within level (novices only, pros only)",
        "",
        "Within a level the level-only baseline is nearly constant, so its Spearman is not "
        "meaningful there; a positive within-level Spearman for a model means it ranks strokes "
        "of the same level.",
        "",
    ]
    for lv in ("novice", "pro"):
        lines += [
            f"**{lv.title()}s: MAE**",
            "",
            md_table(metric_table(scores[lv], "mae", methods)),
            "",
            f"**{lv.title()}s: Spearman**",
            "",
            md_table(metric_table(scores[lv], "spearman", methods)),
            "",
        ]
    pivot = folds.pivot(index="test_player", columns="method", values=["mae", "spearman"]).round(3)
    pivot.columns = [f"{metric} {m}" for metric, m in pivot.columns]
    lines += [
        "## Per test player (score_mean; mean over seeds; Spearman n/a if constant)",
        "",
        md_table(pivot.fillna("n/a")),
        "",
        "Figure: `quality_scatter.png`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
