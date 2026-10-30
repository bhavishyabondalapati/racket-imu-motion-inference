# Design decisions

Each entry: what was chosen, the alternatives considered, and why. Newest milestone last.

## M0: Scaffold

### Python 3.11 in a per-project conda env
- **Chosen:** conda env `racket-imu-motion-inference` with Python 3.11, packages installed with pip.
- **Alternatives:** venv; uv; Python 3.12.
- **Why:** the spec fixes 3.11, and every pinned library ships wheels for it on both macOS arm64 and Linux (checked for mediapipe and torch before writing CI). Conda matches the owner's setup. The env is named after the project rather than the session folder so its purpose is obvious.

### Exact pins in `pyproject.toml`; `requirements.txt` only points to it
- **Chosen:** `==` pins in `pyproject.toml`, taken from what pip resolved together on 2026-10-30. `requirements.txt` contains `-e .[dev]`.
- **Alternatives:** loose ranges (`>=`); a lock file tool (pip-tools, uv lock); duplicating pins in both files.
- **Why:** results must be reproducible, and ranges let versions drift silently. A full lock file is heavier than this project needs. Keeping pins in one place avoids two lists going out of sync.
- **Note:** only top-level packages are pinned; their sub-dependencies are not. Acceptable for now; revisit if CI and local runs disagree.

### `pyarrow` added beyond the spec's list
- **Chosen:** pin `pyarrow`.
- **Why:** M3 writes `strokes.parquet`, and pandas needs pyarrow (or fastparquet) to do that. pyarrow is the pandas default.

### MediaPipe version and API
- **Chosen:** `mediapipe==1.2.0`.
- **Observed:** this version has only the `mediapipe.tasks` API; the old `mp.solutions.pose` API is gone. M4 will use the Tasks `PoseLandmarker`, which needs a downloaded `.task` model file. Model files are git-ignored and will be downloaded by a script with a pinned URL.

### `src/` layout, installed in editable mode
- **Alternatives:** flat package at the repo root.
- **Why:** with `src/`, tests import the installed package, not whatever happens to be in the current folder, which catches packaging mistakes early.

### One YAML config, deep-merged overrides
- **Chosen:** `configs/default.yaml` holds every parameter. `load_config(path, overrides)` merges an optional experiment file and a dict on top. Lists are replaced, not merged.
- **Alternatives:** Hydra / OmegaConf; Python dataclasses; argparse flags.
- **Why:** plain YAML plus about 40 lines of code is easy to read and explain, and a run can save the exact merged dict next to its results. Hydra is powerful but adds a lot of concepts for little gain here.
- **Initial values:** thresholds marked "initial" in the YAML (sync search window, gyro peak threshold, stroke spacing, visibility cut-offs, filter gains) are guesses made before seeing any data. Each will be checked against real recordings in its milestone and any change logged here.

### Seeding
- **Chosen:** `set_seed(seed)` seeds `random`, NumPy's global state, and torch (which covers MPS), and returns a `numpy.random.Generator`.
- **Why:** new code should pass the returned Generator explicitly; global seeding covers libraries that use global state. Full bitwise determinism on MPS is not guaranteed by PyTorch, which is why the deep models are reported as mean ± std over 3 seeds.

### Device selection
- **Chosen:** `get_device("auto")` returns `mps` on Apple Silicon, else `cpu`.
- **Why:** the dev machine is an M4 Pro; CI and other machines fall back to CPU. Latency (M7) is measured on CPU as the spec requires.

### Checkpoints and large files
- **Problem:** the spec asks to ignore checkpoints over 5 MB, but `.gitignore` cannot filter by size.
- **Chosen:** ignore all common checkpoint formats (`*.pt`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.joblib`, `*.pkl`), and add `scripts/check_large_files.py`, run by `make lint` and CI, which fails if any tracked file is over 5 MB.
- **Alternatives:** Git LFS; a pre-commit hook.
- **Why:** no checkpoint needs to be in the repo, since results are regenerated. The size check also protects against accidentally committed data or media of any type. A small checkpoint can still be added on purpose with `git add -f`.

### Makefile placeholders fail on purpose
- **Chosen:** `make sync` … `make report` print "not implemented yet (Mx)" and exit 1.
- **Alternative:** placeholders that just print and succeed.
- **Why:** a placeholder that succeeds would let `make all` look green with no work done. That would quietly break the rule against reporting results that don't exist.

### CI installs CPU-only torch
- **Chosen:** CI first installs the pinned torch version from PyTorch's CPU wheel index, then installs everything else.
- **Why:** the default Linux torch wheel pulls in about 2 GB of CUDA libraries that a CPU runner cannot use. The version stays the same; only the build differs (`+cpu`).
