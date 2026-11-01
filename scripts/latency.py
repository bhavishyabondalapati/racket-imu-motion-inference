"""D8: per-window inference latency on the laptop CPU. Writes results/<run_id>/. `make latency`.

Usage:
    python scripts/latency.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from racketimu.config import PROJECT_ROOT, load_config
from racketimu.infer import infer, load_models, stroke_to_window
from racketimu.io import load_dataset
from racketimu.runinfo import git_state, new_run_dir, provenance_line, write_run_files


def cpu_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out or platform.processor()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    try:
        models = load_models()
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    git, start = git_state(), time.time()
    ds = load_dataset(PROJECT_ROOT / cfg["dataset"]["root"], cfg["dataset"])
    windows = [stroke_to_window(ds.strokes[i]) for i in ds.included()["sample_id"]]
    n_runs, warmup = int(cfg["latency"]["n_runs"]), 20
    for w in windows[:warmup]:
        infer(w, models.train_rate_hz, models)
    times = []
    for k in range(n_runs):
        w = windows[k % len(windows)]
        t0 = time.perf_counter()
        infer(w, models.train_rate_hz, models)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    out = new_run_dir(cfg, "d8")
    info = write_run_files(out, cfg, git, time.time() - start)
    result = {
        "n_runs": n_runs,
        "warmup_runs": warmup,
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_ms": float(t.mean()),
        "max_ms": float(t.max()),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "what_is_timed": "infer(): angle encoding, windowing, 47 handcrafted features, "
        "RF classifier (300 trees) and 6 RF regressors (300 trees each), "
        "single thread, one stroke at a time",
    }
    (out / "latency.json").write_text(json.dumps(result, indent=2))
    lines = [
        f"# D8 inference latency: {out.name}",
        "",
        provenance_line(info),
        "",
        f"- Hardware: {result['cpu']}, {result['logical_cpus']} logical CPUs, CPU only.",
        f"- Timed: {result['what_is_timed']}.",
        f"- {n_runs} timed runs over real strokes (after {warmup} warm-up runs).",
        f"- **Median {result['median_ms']:.2f} ms, p95 {result['p95_ms']:.2f} ms** per window "
        f"(mean {result['mean_ms']:.2f}, max {result['max_ms']:.2f}).",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
