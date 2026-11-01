# Racket IMU Motion Inference

[![tests](https://github.com/bhavishyabondalapati/racket-imu-motion-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/bhavishyabondalapati/racket-imu-motion-inference/actions/workflows/tests.yml)

This project takes a public dataset of table tennis forehands, recorded with a single motion sensor (IMU) on the racket, and measures what can be inferred reliably from that sensor alone: the stroke type (Basic, Topspin, Push), the coaches' quality score, and whether the player is a novice or a professional. Every result is measured on a player the model has never seen (leave-one-player-out), and a sparsity ablation shows how much sensor data is actually needed and where the predictions fail.

## Dataset and credit

> Sahar S. Tabrizi, Vajiheh Javani, Saeid Pashazadeh. *Data Acquired by a Single Object Sensor (IMU) of Table Tennis Forehand Strokes.* Mendeley Data, Version 3, 2020. DOI: [10.17632/b7bc9y232m.3](https://doi.org/10.17632/b7bc9y232m.3). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The data and its original analysis belong to these authors. This project's contribution is its own evaluation protocol (leave-one-player-out), analysis, ablations, and inference interface. It makes no claim of novelty over the original work. The raw data is not redistributed here: download it from Mendeley Data (see Setup).

> **Status: D3, D6 and D4 done; D5, D7, D8, D9 in progress.** Full results will be summarised here in D9. The data itself is described in [docs/DATASET.md](docs/DATASET.md): 1344 of the authors' 1570 cleaned samples are usable, from 15 players.

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
data/processed/                  derived tables, e.g. strokes.parquet (git-ignored, rebuilt by `make preprocess`)
src/racketimu/__init__.py        package marker and version
src/racketimu/config.py          loads the YAML config and applies overrides
src/racketimu/seed.py            seeds Python/NumPy/torch; picks the device (cpu by default)
src/racketimu/io.py              loads the raw stroke files, converts units, attaches labels/scores, marks exclusions
src/racketimu/datacard.py        builds docs/DATASET.md from the loaded data
src/racketimu/dataset.py         fixed-length windows around the peak gyro row, sin/cos angles, model arrays
src/racketimu/plots.py           review figures (example strokes, class profiles, window coverage)
src/racketimu/assets.py          downloads a file and verifies its SHA-256 hash
scripts/check_large_files.py     fails if any tracked file is over 5 MB
scripts/inspect_dataset.py       writes docs/DATASET.md (`make inspect`)
scripts/preprocess.py            writes data/processed/strokes.parquet and docs/figures/d2/ (`make preprocess`)
tests/test_config.py             tests for config loading and merging
tests/test_seed.py               tests that seeding is reproducible and cpu is the default device
tests/test_assets.py             tests for hash-checked downloads (local files, no network)
tests/conftest.py                builds a tiny synthetic copy of the dataset's file layout
tests/test_io.py                 tests for the loader: timestamps, labels, units, exclusion rules
tests/test_datacard.py           tests for the dataset card builder
tests/test_dataset.py            tests for windowing, padding/cropping, and angle encoding
docs/DESIGN.md                   every design decision, alternatives, and reasons
docs/DATASET.md                  dataset card: counts, players, scores, sampling rate, every exclusion (generated)
docs/figures/d2/                 D2 review figures: example strokes per class, class profiles, window coverage
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
make inspect          # D1: load the dataset and regenerate docs/DATASET.md
make preprocess       # D2: build data/processed/strokes.parquet and the review figures
make help             # list all steps
```

The remaining pipeline steps (`make preprocess`, … `make report`, `make all`) exist as placeholders and exit with an error until their milestone is built.

### How to reproduce (once the pipeline is built)

1. Download the dataset into `data/raw/tabrizi2020/` (see Setup).
2. Run `make all`, then read `results/SUMMARY.md`.

## How it was built

1. **M0, scaffold:** wrote the spec, created the conda env, installed the libraries and pinned the exact versions that resolved together, then added the config loader, seed utility, tests, Makefile, large-file guard, and CI. Then made CPU the default device.

2. **Switch to a public dataset:** replaced the self-recorded tennis plan with the Tabrizi et al. (2020) table tennis dataset and milestones D1 to D9. The Makefile placeholders were updated to match. The old plan's code was tagged `recording-plan-archive`, then removed: opencv-python, mediapipe, ahrs, the pose-model download, its config sections, and `data/sessions.yaml`.

3. **D1, data inspection:** inspected every file before writing code and reported the findings: about 8 rows per second instead of the stated 100 Hz, 223 cleaned samples with no accelerometer or gyroscope data, 5 score criteria (not 3 coaches), and several label spellings. With the owner's decisions, wrote the loader (`io.py`) and the generated dataset card (`make inspect`).

4. **D2, preprocessing:** each stroke file becomes one sample of 48 rows with its peak rotation rate at row 16, with a mask for padding. Pitch is encoded as sin/cos because it wraps at ±180°. Heading channels (yaw, magnetometer) are kept out of the main input to avoid a shortcut.

5. **D3, stroke type:** majority baseline, random forests (including shortcut checks), a small CNN and a GRU, all under 15-fold leave-one-player-out with 3 seeds. A reproduction check found and fixed a seeding bug before the final run.

6. **D6, sparsity ablation:** the same protocol rerun with fewer rows, fewer sensors, shorter windows, and with each sample's average removed. Only removing the static orientation (or keeping only the gyroscope) hurt.

7. **D4, stroke quality:** predicted each coach criterion and the mean against three baselines. Neither model beat the level-only baseline, so the models do not judge quality beyond the player's level.

Later milestones (D5 skill → D9 report) will be added here as they are completed.

## Key concepts

- **Leave-one-player-out (LOPO) cross-validation:** train on all players but one, test on the one left out, repeat for each player. Strokes from one player look alike, so a random split would leak a player's style into training and overstate accuracy. LOPO measures what matters: performance on a new person. Here it has 15 folds, because one player (P06) has no usable accelerometer or gyroscope data.
- **Inspect before modelling:** the dataset description said 100 Hz and 1570 samples; the files showed about 8 rows per second, and 223 of those samples have no accelerometer or gyroscope signal. Building on the description alone would have produced misleading results.
- **Shortcut learning:** a model can score well by picking up something that differs between recording sessions (here, absolute heading) instead of the stroke itself. Keeping heading out of the main input, and testing with and without it, guards against that.
- **Angle wrap-around:** an angle like pitch jumps from +179° to −179° when it crosses ±180°. Encoding it as sin and cos keeps the input continuous.
- **Pooled out-of-fold metrics:** predictions from every test fold are combined before computing the confusion matrix and macro-F1. Several players have only one or two stroke classes, so a per-player macro-F1 would be undefined or misleading.
- **Reproducibility:** exact dependency pins, one config file for all parameters, and one seed function mean anyone can rerun the pipeline and get the same numbers.
- **Config over constants:** thresholds live in `configs/default.yaml`, not in code, so experiments change a file, not the source, and each run saves the config it used.
- **CPU for reported results:** GPU backends like MPS can give slightly different numbers run to run. Reported numbers use CPU so they can be reproduced exactly.
- **CI:** GitHub Actions runs lint and tests on every push, so a broken change is caught immediately.
