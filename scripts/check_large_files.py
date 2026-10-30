"""Fail if any git-tracked file is larger than a limit (default 5 MB).

.gitignore cannot filter by file size, so this catches large checkpoints, data
files, or media that slipped past it. Run by `make lint` and in CI.

Usage:
    python scripts/check_large_files.py [--max-mb 5]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-mb", type=float, default=5.0)
    args = parser.parse_args()

    limit = args.max_mb * 1024 * 1024
    too_big = [(p, p.stat().st_size) for p in tracked_files() if p.exists()]
    too_big = [(p, size) for p, size in too_big if size > limit]
    for path, size in too_big:
        print(f"Too large ({size / 1024 / 1024:.1f} MB > {args.max_mb} MB): {path}")
    if too_big:
        return 1
    print(f"OK: no tracked file is over {args.max_mb} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
