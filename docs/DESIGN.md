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

### `pyarrow` added beyond the spec's list (approved by the owner)
- **Chosen:** pin `pyarrow==26.0.0`.
- **Alternatives:** fastparquet; saving CSV instead of parquet.
- **Why:** M3 writes `strokes.parquet`, and pandas needs pyarrow or fastparquet for that. pyarrow is the engine pandas uses by default and the most widely used. CSV would lose dtypes and is larger.

### ffmpeg from conda-forge, inside the project env
- **Chosen:** `conda install -c conda-forge ffmpeg` into `racket-imu-motion-inference` (installed: ffmpeg 7.1).
- **Alternatives:** Homebrew (system-wide); a pip wrapper such as `imageio-ffmpeg`.
- **Why:** ffmpeg is a binary, not a Python package, so pip can't pin it cleanly. A system-wide install isn't tied to the project. With conda it lives and dies with the env, and installing it left every pip-pinned package unchanged (checked after install).

### MediaPipe: Tasks `PoseLandmarker`, heavy model, pinned by hash
- **API:** `mediapipe==1.2.0` has only the `mediapipe.tasks` API; the old `mp.solutions.pose` is gone. M4 uses `vision.PoseLandmarker`, which needs a separate `.task` model file. Checked that the model loads and runs with this version.
- **Variant chosen: heavy** (float16, version 1, about 30 MB).
- **Alternatives:** `lite` (fastest, least accurate) and `full` (middle ground).
- **Why heavy:** pose here is the *reference* every wrist prediction is scored against, so its error sets a floor on what M6 can show. Pose runs offline, once per video, so speed barely matters: a few minutes of 60 fps video is a few thousand frames. Fast swings blur the wrist, which is where a bigger model should help most. If heavy proves too slow, `full` is a one-line config change, but its reference noise must then be measured again in M4.
- **Pinning:** `configs/default.yaml` stores the variant, a URL with an explicit version (`/float16/1/`, not `/latest/`), and the SHA-256 (`64437af8…cbc7b`). `make models` (`scripts/download_pose_model.py` → `racketimu.assets.download_verified`) downloads to the git-ignored `models/` folder. The file is written as `.part` and renamed only after the hash matches, so a bad download never looks valid. An existing file with the wrong hash is reported, not overwritten.
- **Why hash it:** if Google ever replaced the file at that URL, the reference trajectories would change silently. With the hash check, the download fails loudly instead.

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
- **Why:** new code should pass the returned Generator explicitly; global seeding covers libraries that use global state. Reporting deep models as mean ± std over 3 seeds shows how much results depend on the seed, separately from the device question below.

### Device selection: CPU by default (owner's decision)
- **Chosen:** `train.device: cpu` in the config, and `get_device()` defaults to `cpu`. Every reported experiment (training, evaluation, latency) runs on CPU. `mps` and `auto` (MPS if available, else CPU) are opt-in flags for quick local iteration only. Any other value is rejected, and asking for `mps` where it isn't available is an error, not a silent fallback.
- **Alternative:** default to `mps` on the M4 Pro for speed.
- **Why:** MPS results are **not bit-for-bit reproducible**. Some MPS kernels are non-deterministic, and PyTorch has no deterministic mode that covers them, so the same seed can give slightly different weights and metrics. CPU runs with a fixed seed repeat exactly, which matters more than speed for models this small. Results from an `mps` run must never be copied into `results/` or the README.

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
