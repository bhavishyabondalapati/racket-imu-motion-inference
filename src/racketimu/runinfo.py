"""Provenance for every results folder: git state (captured before the run), hardware, versions."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from racketimu.config import PROJECT_ROOT


def git_state() -> dict:
    """Commit hash and any uncommitted files. Call this before a run starts."""

    def run(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout

    changed = [line[3:] for line in run("status", "--porcelain").splitlines() if line.strip()]
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(changed), "changed": changed}


def environment() -> dict:
    return {
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


def new_run_dir(cfg: dict, prefix: str) -> Path:
    out = PROJECT_ROOT / cfg["paths"]["results"] / f"{prefix}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    return out


def write_run_files(out: Path, cfg: dict, git: dict, seconds: float) -> dict:
    """Write config.yaml and run.json; return the run info."""
    info = {
        **git,
        "duration_s": round(seconds, 1),
        "device": cfg["train"]["device"],
        **environment(),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (out / "run.json").write_text(json.dumps(info, indent=2))
    return info


def provenance_line(info: dict) -> str:
    dirty = f" (uncommitted: {', '.join(info['changed'])})" if info["dirty"] else ""
    return (
        f"Commit `{info['commit'][:10]}`{dirty}; device {info['device']}; "
        f"{info['processor'] or info['machine']} ({info['system']}); {info['duration_s']} s."
    )
