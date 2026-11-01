"""Load the dataset and write docs/DATASET.md (D1). Run with `make inspect`.

Usage:
    python scripts/inspect_dataset.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from racketimu.config import PROJECT_ROOT, load_config
from racketimu.datacard import build_dataset_card
from racketimu.io import DatasetFiles, load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = PROJECT_ROOT / cfg["dataset"]["root"]
    try:
        ds = load_dataset(root, cfg["dataset"])
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    out = PROJECT_ROOT / cfg["paths"]["docs"] / "DATASET.md"
    out.write_text(build_dataset_card(ds, DatasetFiles(root), cfg))
    inc = ds.included()
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}: {len(inc)} of {len(ds.index)} samples used, ")
    print(f"{inc['player'].nunique()} players.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
