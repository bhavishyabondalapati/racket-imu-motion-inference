"""D5: skill level and cross-group generalization. Writes results/<run_id>/. `make skill`.

Usage:
    python scripts/skill.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from racketimu import skill as S
from racketimu.classify import prepare_inputs
from racketimu.config import PROJECT_ROOT, load_config
from racketimu.datacard import md_table
from racketimu.runinfo import git_state, new_run_dir, provenance_line, write_run_files


def pm(s: pd.Series, fmt: str = ".3f") -> str:
    return f"{s.mean():{fmt}}" if len(s) == 1 else f"{s.mean():{fmt}} ± {s.std():{fmt}}"


def level_table(metrics: pd.DataFrame) -> pd.DataFrame:
    g = metrics.groupby("model", sort=False)
    return pd.DataFrame(
        {
            "seeds": g.size(),
            "accuracy": g["accuracy"].apply(pm),
            "balanced accuracy": g["balanced_accuracy"].apply(pm),
            "macro-F1": g["macro_f1"].apply(pm),
        }
    ).rename_axis("model")


def per_player(oof: pd.DataFrame) -> pd.DataFrame:
    acc = oof.assign(ok=oof["y_true"] == oof["y_pred"])
    t = acc.groupby(["test_player", "model"])["ok"].mean().unstack().round(3)
    t.insert(0, "level", acc.groupby("test_player")["y_true"].first().map({0: "novice", 1: "pro"}))
    t.insert(1, "n", acc[acc["model"] == "majority"].groupby("test_player").size())
    return t


def by_stroke(oof: pd.DataFrame) -> pd.DataFrame:
    acc = oof[oof["model"] != "majority"].assign(ok=lambda d: d["y_true"] == d["y_pred"])
    return acc.groupby(["stroke", "model"])["ok"].mean().unstack().round(3)


def cross_table(summary: pd.DataFrame) -> pd.DataFrame:
    g = summary.groupby("setting", sort=False)
    return pd.DataFrame(
        {
            "mode": g["mode"].first(),
            "test samples": g["n_test"].first(),
            "stroke-type macro-F1": g["type_macro_f1"].apply(pm),
            "score_mean MAE": g["mae_score_mean"].apply(lambda s: pm(s, ".2f")),
            "score_mean Spearman": g["spearman_score_mean"].apply(pm),
        }
    ).rename_axis("setting")


def drop_lines(summary: pd.DataFrame) -> list[str]:
    lines = []
    for level in S.LEVELS:
        other = "pro" if level == "novice" else "novice"
        w = summary[(summary["test_level"] == level) & (summary["mode"] == "within-group LOPO")]
        c = summary[(summary["test_level"] == level) & (summary["mode"] == "cross-group")]
        lines.append(
            f"- **Testing on {level}s, training on {other}s instead of other {level}s:** "
            f"stroke-type macro-F1 {c['type_macro_f1'].mean() - w['type_macro_f1'].mean():+.3f}; "
            f"score_mean MAE "
            f"{c['mae_score_mean'].mean() - w['mae_score_mean'].mean():+.2f} points; "
            f"Spearman {c['spearman_score_mean'].mean() - w['spearman_score_mean'].mean():+.3f}."
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
    inp = prepare_inputs(pd.read_parquet(path), cfg)
    level_oof = S.run_level_classification(inp, cfg)
    type_oof, score_oof = S.run_cross_group(inp, cfg)
    out = new_run_dir(cfg, "d5")
    info = write_run_files(out, cfg, git, time.time() - start)
    level_oof.to_csv(out / "level_oof.csv.gz", index=False)
    type_oof.to_csv(out / "cross_group_type_oof.csv.gz", index=False)
    score_oof.to_csv(out / "cross_group_quality_oof.csv.gz", index=False)
    metrics, summary = S.level_metrics(level_oof), S.cross_group_summary(type_oof, score_oof)
    (out / "metrics.json").write_text(
        json.dumps(
            {"level": metrics.to_dict("records"), "cross_group": summary.to_dict("records")},
            indent=2,
            default=float,
        )
    )
    lines = [
        f"# D5 skill level and cross-group generalization: {out.name}",
        "",
        provenance_line(info),
        "",
        "## Part 1: novice vs pro (15-fold LOPO, D3 folds)",
        "",
        "Each test player has a single level, so results are pooled out-of-fold metrics "
        "(mean ± std "
        "over 3 seeds) plus per-player accuracy, not per-fold F1.",
        "",
        md_table(level_table(metrics)),
        "",
        "**Per test player: share of strokes assigned the correct level (mean over seeds)**",
        "",
        md_table(per_player(level_oof)),
        "",
        "**Accuracy by stroke type (mean over seeds and folds)**",
        "",
        md_table(by_stroke(level_oof)),
        "",
        "## Part 2: cross-group generalization (RF only)",
        "",
        "Within-group LOPO trains on the other players of the same level; cross-group trains "
        "on all players of the other level. Validation players (2) always come from the "
        "training group; the "
        "forests are fitted on the fit set only, as in D3/D4. Pooled over test samples, mean ± std "
        "over 3 seeds. Stroke-type macro-F1 averages the classes present in the test group.",
        "",
        md_table(cross_table(summary)),
        "",
        "**Change from within-group LOPO to cross-group (negative F1 / positive MAE = worse):**",
        "",
        *drop_lines(summary),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
