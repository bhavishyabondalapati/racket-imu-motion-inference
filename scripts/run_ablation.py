"""D6: sparsity ablation for stroke-type classification. Writes results/<run_id>/. `make ablation`.

Usage:
    python scripts/run_ablation.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from racketimu.ablation import conditions, run_condition, summarize
from racketimu.config import PROJECT_ROOT, load_config
from racketimu.datacard import md_table
from racketimu.plots import plot_ablation_deltas, plot_f1_vs_stride
from racketimu.runinfo import git_state, new_run_dir, provenance_line, write_run_files


def table(summary: pd.DataFrame, model: str) -> pd.DataFrame:
    s = summary[summary["model"] == model]
    return pd.DataFrame(
        {
            "condition": s["label"],
            "group": s["group"],
            "macro-F1": [
                f"{m:.3f} ± {d:.3f}" for m, d in zip(s["f1_mean"], s["f1_std"], strict=True)
            ],
            "accuracy": s["acc_mean"].map("{:.3f}".format),
            "Δ macro-F1 vs full": [
                f"{m:+.3f} ± {d:.3f}" if c != "full" else "-"
                for m, d, c in zip(s["delta_mean"], s["delta_std"], s["condition"], strict=True)
            ],
        }
    )


def ranking_lines(summary: pd.DataFrame, model: str) -> list[str]:
    s = summary[(summary["model"] == model) & (summary["condition"] != "full")]
    s = s.sort_values("delta_mean")
    return [
        f"{k}. {r.label}: {r.delta_mean:+.3f} macro-F1" for k, r in enumerate(s.itertuples(), 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    path = PROJECT_ROOT / cfg["paths"]["processed"] / "strokes.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run `make preprocess` first.", file=sys.stderr)
        return 1
    git, start = git_state(), time.time()
    windows, deep = pd.read_parquet(path), cfg["ablation"]["deep_model"]
    conds = conditions(cfg)
    oof = pd.concat([run_condition(windows, c, cfg, deep) for c in conds], ignore_index=True)
    out = new_run_dir(cfg, "d6")
    info = write_run_files(out, cfg, git, time.time() - start)
    oof.to_csv(out / "oof_predictions.csv.gz", index=False)
    summary = summarize(oof, conds)
    (out / "metrics.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, default=float)
    )
    plot_f1_vs_stride(summary, cfg, deep, out / "f1_vs_stride.png")
    plot_ablation_deltas(summary[summary["model"] == "rf"], out / "ablation_deltas.png")
    lines = [
        f"# D6 sparsity ablation: {out.name}",
        "",
        provenance_line(info),
        "",
        "Same protocol as D3: 15 LOPO folds (P06 not a test player), 1 novice + 1 pro inner "
        "validation players per fold, CPU, 3 seeds, pooled out-of-fold macro-F1 (mean ± std "
        "over seeds). Δ is the per-seed difference from the same model on full data. "
        f"Full data = primary channels (acc, gyro, roll, pitch sin/cos), every row, "
        f"{cfg['window']['n_rows']}-row window. The RF uses handcrafted features of the "
        f"channels in each condition; the deep model is the D3 {deep.upper()}.",
        "",
        "## Random forest (every condition)",
        "",
        md_table(table(summary, "rf"), index=False),
        "",
        "Reductions ranked by how much they hurt (RF):",
        "",
        *ranking_lines(summary, "rf"),
        "",
        f"## {deep.upper()} (full data, motion-only, every 4th row)",
        "",
        md_table(table(summary, deep), index=False),
        "",
        "Figures: `f1_vs_stride.png`, `ablation_deltas.png`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
