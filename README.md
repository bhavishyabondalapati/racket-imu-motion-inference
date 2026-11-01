# Racket IMU Motion Inference

[![tests](https://github.com/bhavishyabondalapati/racket-imu-motion-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/bhavishyabondalapati/racket-imu-motion-inference/actions/workflows/tests.yml)

This project takes a public dataset of table tennis forehands, recorded with a single motion sensor (IMU) on the racket, and measures what can be inferred reliably from that sensor alone: the stroke type (Basic, Topspin, Push), the coaches' quality score, and whether the player is a novice or a professional. Every result is measured on a player the model has never seen (leave-one-player-out), and a sparsity ablation shows how much sensor data is actually needed and where the predictions fail.

## Dataset and credit

> Sahar S. Tabrizi, Vajiheh Javani, Saeid Pashazadeh. *Data Acquired by a Single Object Sensor (IMU) of Table Tennis Forehand Strokes.* Mendeley Data, Version 3, 2020. DOI: [10.17632/b7bc9y232m.3](https://doi.org/10.17632/b7bc9y232m.3). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The data and its original analysis belong to these authors. This project's contribution is its own evaluation protocol (leave-one-player-out), analysis, ablations, and inference interface. It makes no claim of novelty over the original work. The raw data is not redistributed here: download it from Mendeley Data (see Setup).

> **Status: M0 done; D1 (data inspection) next.** No data has been processed yet, so there are no results.

## Results

_Empty until D9. Every number here will come from `results/SUMMARY.md`, which `make report` generates from runs on the real dataset._

## Tools and libraries

| Library | Why |
|---|---|
| numpy, pandas | Arrays and tables for the IMU streams and stroke metadata |
| scipy | Signal processing: anti-aliased resampling, filtering, peak detection |
| scikit-learn | Random-forest baseline, metrics, and cross-validation helpers |
| torch | Small 1D CNN and GRU models; runs on CPU for reported results (Apple `mps` is opt-in for quick local runs) |
| matplotlib | Stroke plots, reliability diagrams, ablation figures |
| pyyaml | Config file |
| pyarrow | Lets pandas write `strokes.parquet` |
| pytest, ruff | Tests and lint/format, run locally and in GitHub Actions |

Rotation math (D2) uses `scipy.spatial.transform`, so no separate orientation library is needed. The code and dependencies from the earlier self-recorded plan (video sync, MediaPipe pose) are kept under the git tag `recording-plan-archive`.

## File structure

```
CLAUDE.md                        project spec and ground rules
README.md                        this file
pyproject.toml                   package definition with exact dependency pins, ruff and pytest settings
requirements.txt                 installs the package plus dev tools from pyproject.toml
Makefile                         one command per step (test, lint, and D1-D9 placeholders)
.gitignore                       keeps data, videos, checkpoints, secrets, and envs out of git
.github/workflows/tests.yml      CI: lint and tests on every push
configs/default.yaml             every threshold, window size, and seed
data/raw/, data/interim/         raw dataset (data/raw/tabrizi2020/) and intermediate files (git-ignored, local only)
data/processed/                  small derived tables
src/racketimu/__init__.py        package marker and version
src/racketimu/config.py          loads the YAML config and applies overrides
src/racketimu/seed.py            seeds Python/NumPy/torch; picks the device (cpu by default)
src/racketimu/assets.py          downloads a file and verifies its SHA-256 hash
scripts/check_large_files.py     fails if any tracked file is over 5 MB
tests/test_config.py             tests for config loading and merging
tests/test_seed.py               tests that seeding is reproducible and cpu is the default device
tests/test_assets.py             tests for hash-checked downloads (local files, no network)
docs/DESIGN.md                   every design decision, alternatives, and reasons
docs/figures/                    stroke plots and result figures (filled from D2 on)
results/                         one folder per run + SUMMARY.md (filled from D3 on)
```

## Setup

```bash
conda create -n racket-imu-motion-inference python=3.11
conda activate racket-imu-motion-inference
make install          # same as: pip install -r requirements.txt
```

Download the dataset: open https://data.mendeley.com/datasets/b7bc9y232m/3, click "Download All", and unzip it into `data/raw/tabrizi2020/`.

## How to run

```bash
make test             # unit tests (synthetic data only)
make lint             # ruff + large-file check
make help             # list all steps
```

Pipeline steps (`make inspect`, `make preprocess`, … `make report`, `make all`) exist as placeholders and exit with an error until their milestone is built.

### How to reproduce (once the pipeline is built)

1. Download the dataset into `data/raw/tabrizi2020/` (see Setup).
2. Run `make all`, then read `results/SUMMARY.md`.

## How it was built

1. **M0, scaffold:** wrote the spec, created the conda env, installed the libraries and pinned the exact versions that resolved together, then added the config loader, seed utility, tests, Makefile, large-file guard, and CI. Then made CPU the default device.

2. **Switch to a public dataset:** replaced the self-recorded tennis plan with the Tabrizi et al. (2020) table tennis dataset and milestones D1 to D9. The Makefile placeholders were updated to match. The old plan's code was tagged `recording-plan-archive`, then removed: opencv-python, mediapipe, ahrs, the pose-model download, its config sections, and `data/sessions.yaml`.

Later milestones (D1 inspection → D9 report) will be added here as they are completed.

## Key concepts

- **Leave-one-player-out (LOPO) cross-validation:** train on all players but one, test on the one left out, repeat for each player. Strokes from one player look alike, so a random split would leak a player's style into training and overstate accuracy. LOPO measures what matters: performance on a new person.
- **Reproducibility:** exact dependency pins, one config file for all parameters, and one seed function mean anyone can rerun the pipeline and get the same numbers.
- **Config over constants:** thresholds live in `configs/default.yaml`, not in code, so experiments change a file, not the source, and each run saves the config it used.
- **CPU for reported results:** GPU backends like MPS can give slightly different numbers run to run. Reported numbers use CPU so they can be reproduced exactly.
- **CI:** GitHub Actions runs lint and tests on every push, so a broken change is caught immediately.
