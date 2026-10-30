"""Download the MediaPipe PoseLandmarker model named in the config and check its hash.

Usage:
    python scripts/download_pose_model.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from racketimu.assets import download_verified
from racketimu.config import PROJECT_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    model = load_config(args.config)["pose_model"]
    dest = PROJECT_ROOT / model["path"]
    try:
        download_verified(model["url"], dest, model["sha256"])
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {model['variant']} pose model at {dest} (sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
