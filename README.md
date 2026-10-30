# Racket IMU Motion Inference

[![tests](https://github.com/bhavishyabondalapati/racket-imu-motion-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/bhavishyabondalapati/racket-imu-motion-inference/actions/workflows/tests.yml)

A smartphone taped to a tennis racket records how the racket accelerates and rotates. This project uses only that signal to (1) find each stroke, (2) say whether it was a forehand, backhand, or serve, and (3) predict the path of the hitting wrist, checking every prediction against pose tracked from a synchronized video. The goal is to measure honestly how much movement one racket sensor can reveal about a player the model has never seen, and where it fails.

> **Status: M0 (scaffold).** No recordings have been processed yet, so there are no results.

## Results

_Empty until M8. Every number here will come from `results/SUMMARY.md`, which is generated from real recordings by `make report`._

## Tools and libraries

| Library | Why |
|---|---|
| numpy, pandas | Arrays and tables for the IMU streams and stroke metadata |
| scipy | Signal processing: resampling, filtering, peak detection for taps and strokes |
| scikit-learn | Random-forest baseline, metrics, and cross-validation helpers |
| torch | Small 1D CNN and GRU models; uses Apple Silicon's `mps` device locally |
| opencv-python | Reading video frames and writing replay videos |
| mediapipe | Pose estimation on video, which gives the reference wrist trajectory |
| ahrs | Ready-made, tested Madgwick orientation filter |
| matplotlib | Verification plots (sync, segmentation, reliability diagrams) |
| pyyaml | Config and session metadata files |
| pyarrow | Lets pandas write `strokes.parquet` |
| pytest, ruff | Tests and lint/format, run locally and in GitHub Actions |

ffmpeg (for extracting audio for sync) will be added in M1.

## File structure

```
CLAUDE.md                        project spec and ground rules
README.md                        this file
pyproject.toml                   package definition with exact dependency pins, ruff and pytest settings
requirements.txt                 installs the package plus dev tools from pyproject.toml
Makefile                         one command per step (test, lint, and pipeline placeholders)
.gitignore                       keeps data, videos, checkpoints, secrets, and envs out of git
.github/workflows/tests.yml      CI: lint and tests on every push
configs/default.yaml             every threshold, window size, and seed
data/sessions.yaml               recording metadata (players, sessions); empty template for now
data/raw/, data/interim/         recordings and intermediate files (git-ignored, local only)
data/processed/                  small derived tables
src/racketimu/__init__.py        package marker and version
src/racketimu/config.py          loads the YAML config and applies overrides
src/racketimu/seed.py            seeds Python/NumPy/torch; picks mps or cpu
scripts/check_large_files.py     fails if any tracked file is over 5 MB
tests/test_config.py             tests for config loading and merging
tests/test_seed.py               tests that seeding is reproducible
docs/DESIGN.md                   every design decision, alternatives, and reasons
docs/figures/                    verification plots (filled from M1 on)
results/                         one folder per run + SUMMARY.md (filled from M5 on)
```

## Setup

```bash
conda create -n racket-imu-motion-inference python=3.11
conda activate racket-imu-motion-inference
make install          # same as: pip install -r requirements.txt
```

## How to run

```bash
make test             # unit tests (synthetic data only)
make lint             # ruff + large-file check
make help             # list all steps
```

Pipeline steps (`make sync`, `make frames`, … `make report`, `make all`) exist as placeholders and exit with an error until their milestone is built.

### How to reproduce (once data exists)

1. Record sessions following the protocol in `CLAUDE.md` and put each one under `data/raw/<session_id>/`.
2. Describe each session in `data/sessions.yaml`.
3. Run `make all`, then read `results/SUMMARY.md`.

## How it was built

1. **M0, scaffold:** wrote the spec, created the conda env, installed the libraries and pinned the exact versions that resolved together, then added the config loader, seed utility, tests, Makefile, large-file guard, and CI.

Later milestones (M1 sync → M8 report) will be added here as they are completed.

## Key concepts

- **Leave-one-player-out (LOPO) cross-validation:** train on all players but one, test on the one left out, repeat for each player. Strokes from one player look alike, so a random split would leak a player's style into training and overstate accuracy. LOPO measures what matters: performance on a new person.
- **Reproducibility:** exact dependency pins, one config file for all parameters, and one seed function mean anyone can rerun the pipeline and get the same numbers.
- **Config over constants:** thresholds live in `configs/default.yaml`, not in code, so experiments change a file, not the source, and each run saves the config it used.
- **CI:** GitHub Actions runs lint and tests on every push, so a broken change is caught immediately.
