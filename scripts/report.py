"""D9: write results/SUMMARY.md from the saved result files. Run with `make report`.

Usage:
    python scripts/report.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from racketimu.config import PROJECT_ROOT, load_config
from racketimu.report import build_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    results = PROJECT_ROOT / cfg["paths"]["results"]
    out = results / "SUMMARY.md"
    out.write_text(build_summary(results))
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
