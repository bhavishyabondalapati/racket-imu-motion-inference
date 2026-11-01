"""D9: write results/SUMMARY.md from the saved result files. Run with `make report`.

Usage:
    python scripts/report.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from racketimu.config import PROJECT_ROOT, load_config
from racketimu.plots import plot_ablation_deltas, plot_f1_vs_stride
from racketimu.report import build_summary, latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    results = PROJECT_ROOT / cfg["paths"]["results"]
    out = results / "SUMMARY.md"
    out.write_text(build_summary(results))
    d6 = latest(results, "d6")
    if d6 is not None:  # redraw D6 figures from the saved metrics (labels fixed after the run)
        summary = pd.DataFrame(json.loads((d6 / "metrics.json").read_text()))
        summary["label"] = summary["label"].str.replace("2th", "2nd")
        figs = PROJECT_ROOT / cfg["paths"]["figures"] / "d6"
        plot_f1_vs_stride(summary, cfg, cfg["ablation"]["deep_model"], figs / "f1_vs_stride.png")
        plot_ablation_deltas(summary[summary["model"] == "rf"], figs / "ablation_deltas.png")
        print(
            f"Redrew D6 figures from {d6.name}/metrics.json into {figs.relative_to(PROJECT_ROOT)}/"
        )
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
