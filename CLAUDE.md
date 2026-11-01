# CLAUDE.md: Racket IMU Motion Inference (public dataset version)

## Project goal

Using a public dataset recorded with a single IMU mounted on a racket, determine what can be inferred reliably about a stroke from sparse inertial data alone, and where it fails. This mirrors the problem in the Opus 133 FramePoint co-op posting (infer movement from limited equipment-mounted sensor data, evaluate on unseen players, report uncertainty and failure cases, deliver a documented inference interface).

Three tasks, in order of priority:

1. Stroke type classification: Basic, Topspin, Push forehand.
2. Stroke quality estimation: predict the coaches' quality scores from the IMU signal.
3. Skill level: novice vs professional, including cross-group generalization (train on one group, test on the other).

Plus a sparsity ablation (how much sensor data is actually needed), which is the most important analysis for this audience.

The headline protocol is leave-one-player-out (LOPO) cross-validation: every reported result is on a player the model has never seen.

Owner: Bhavishya Bondalapati. Package: `racketimu`. Repo: `racket-imu-motion-inference`.

## Dataset

- Title: Data Acquired by a Single Object Sensor (IMU) of Table Tennis Forehand Strokes
- Authors: Sahar S. Tabrizi, Vajiheh Javani, Saeid Pashazadeh (University of Tabriz)
- Source: Mendeley Data, Version 3, 29 October 2020, DOI: 10.17632/b7bc9y232m.3
- Licence: CC BY 4.0 (reuse allowed with attribution)
- Described contents: 16 players (8 novice, 8 professional), one 9-axis IMU (BNO055: accelerometer, gyroscope, magnetometer, plus Euler angles) mounted on the racket face, 1570 forehand samples of three types, labeled and scored manually by 3 coaches.
- Related papers: the dataset's Data in Brief article (2020) and the authors' IEEE paper (https://ieeexplore.ieee.org/abstract/document/9127525).

The owner downloads the files manually into `data/raw/tabrizi2020/`. Raw data is git-ignored.

### Facts established in D1 (from the files and the authors' papers)

- One raw CSV per stroke (`Raw/Sensory-Data/<id>.csv`, ids 1 to 1728). The `Processed/` files are the same rows zero-padded at the front to 70 rows.
- C1 to C5 are five evaluation criteria. Each published value is the average of the 3 coaches; individual coach scores are not published, so agreement between coaches cannot be computed.
- Strokes are shadow-play (no ball).
- Stated sampling rate: 100 Hz (Data in Brief article) and 70 Hz (2021 Computational Intelligence and Neuroscience paper). Observed: about 9 rows per second. Report all three; do not speculate about the cause.
- The 2021 paper's tables list 1525 samples per criterion, versus 1570 in the authors' cleaned set.
- The Euler angle order is not documented.
- Units (BNO055 defaults, checked against the data): acc raw/100 = m/s², gyro raw/16 = deg/s, mag raw/16 = µT, Euler angles in degrees.

### Sample set and evaluation decisions (owner-approved in D1)

- Start from the authors' cleaned ID set (the 1570 processed files) and load the rows from the Raw files. Do not re-add the 157 ids the authors removed.
- Exclude the 3 empty raw files and the 223 files whose accelerometer and gyroscope are zero in every row, leaving 1344 samples. List every exclusion with its reason in `docs/DATASET.md`.
- Keep exact duplicate rows, and report the duplicate rate.
- Sample 720: follow the spreadsheet (player P05), not the scores CSV (P06), and note it.
- Any model that uses accelerometer or gyroscope data uses 15 LOPO folds: P06 is not used as a test player. (Its only usable sample was ID 720, which the spreadsheet assigns to P05, so P06 has no usable acc/gyro samples at all.) Report which stroke classes each player has (e.g. P01 has basics only).
- Pool out-of-fold predictions for the overall confusion matrix and macro-F1, and report per-fold results alongside.

## Ground rules (non-negotiable)

1. Never fabricate data or results. Every number in `results/`, the README, or any summary must come from a script run on the real dataset. Synthetic data is allowed only in `tests/`.
2. Never invent prior-work numbers. Do not state what the original authors achieved unless the owner provides the text of their paper. If comparison is not possible, say so.
3. Do not claim novelty over the original authors. The dataset and its original analysis are theirs. This project's contribution is its own evaluation protocol, analysis, ablations, and inference interface. Say this clearly in the README.
4. Attribution: the README must cite the dataset (authors, title, Mendeley Data, DOI, CC BY 4.0) near the top.
5. Inspect before assuming. Before writing any loader, inspect the real files (file list, headers, units, sampling rate, how strokes, labels, players, and scores are linked) and report back. Wait for the owner's OK.
6. No leakage. No sample from the test player may influence training, feature scaling, hyperparameter choice, or calibration.
7. Document every design decision in `docs/DESIGN.md` (choice, alternatives, reason). The owner must be able to defend each one in an interview.
8. Small, clear commits. Commit after each logical step, push at the end of each milestone, never force-push.
9. Ask the owner before: changing the evaluation protocol, adding heavy dependencies, or skipping an acceptance criterion.
10. Stop at the end of each milestone and summarize: what was built, what to check (including figures), open questions.

## Milestones

M0 (scaffold, CI, config, seeding, CPU-default device) is complete and stays as is. The earlier recording-based M1 to M4 are replaced by the milestones below. The video, sync, and pose-reference code is not needed; remove its placeholders from the Makefile and note the change in DESIGN.md.

### D1: Data inspection and dataset card

- Inspect `data/raw/tabrizi2020/` and report: file list, columns, units, sampling rate (mean and jitter), how strokes are delimited, how labels, coach scores, and player levels are stored, and any missing or inconsistent data.
- Wait for the owner's OK, then write the loader.
- Generate `docs/DATASET.md` with `make inspect`: players, samples per class, per player, and per level, which stroke classes each player has, score distributions, sampling rate (stated vs observed), duplicate rate, and every exclusion with its reason. Agreement between coaches cannot be computed (only coach-averaged scores are published); say so instead.
- Accept when: sample counts are reconciled with the dataset description (1570 samples, 16 players) and the 2021 paper (1525), or every difference is explained.

### D2: Preprocessing

- Strokes are already one per file, so no segmentation. At about 9 rows per second there is no peak-based time window: each stroke file is one sample, padded or cropped to a fixed number of rows (configurable, default 32) centered on the row with peak gyro magnitude, with a mask marking real rows.
- No world-frame or gravity-removal transform, because the Euler angle order is not documented. Inputs are sensor-frame acceleration and angular velocity (converted units) and the Euler angles as separate channels. Record this in DESIGN.md.
- Save `data/processed/strokes.parquet`. Commit it only if it contains no personal data; otherwise git-ignore it.
- Accept when: plots of a few strokes per class look sensible (saved to `docs/figures/`).

### D3: Stroke type classification

- Baselines: majority class, then a random forest on handcrafted features (per-axis statistics, peak angular velocity, orientation change, window energy).
- Deep models: a small 1D CNN and a small GRU in PyTorch, 3 seeds, reported as mean ± std.
- Protocol: outer LOPO (15 folds; P06 is not a test player), with an inner validation split by player for early stopping and temperature scaling.
- Metrics: accuracy, macro-F1, confusion matrix (pooled out-of-fold predictions), per-fold results, per-player and per-level (novice vs pro) breakdowns.
- Calibration: expected calibration error before and after temperature scaling, plus a reliability diagram.
- Accept when: results are saved under `results/<run_id>/` with the config, commit hash, and metrics JSON.

### D4: Stroke quality estimation

- Predict each criterion C1 to C5 separately, and their mean, from the IMU window.
- Baselines: the mean training score, the per-class mean score, and the mean training score for the test player's level. Scores depend strongly on level, so the model only counts as judging stroke quality if it beats the level-only baseline.
- Also evaluate within each level (novices only, pros only).
- Models: the random forest regressor and the better of the CNN/GRU, under LOPO.
- Metrics: MAE and Spearman correlation. There is no coach-agreement ceiling: only coach-averaged scores are published, so coach agreement cannot be computed. Say this in the report.
- Accept when: the report states whether the model beats each baseline, in particular the level-only baseline, on unseen players.

### D5: Skill level and cross-group generalization

- Novice vs pro classification under LOPO.
- Cross-group test: train stroke-type and quality models on pros only and test on novices, then the reverse. Report how much performance drops. This shows whether a model built on one population transfers to another, which matters for a consumer product.
- Accept when: the table shows within-group LOPO results next to the cross-group results.

### D6: Sparsity ablation

Re-run the best stroke-type model and the best quality model under LOPO with:

- sampling rate: keep every row, every 2nd row, and every 4th row (native is about 9 rows per second)
- sensors: accelerometer only, gyroscope only, accelerometer + gyroscope, all including magnetometer and orientation
- window length: 100%, 75%, 50% of default

Report each condition against the full-data result, mean ± std over seeds, and plot performance against sampling rate. State which reductions hurt most.

- Accept when: `results/` contains the ablation table and figures.

### D7: Error analysis

- The 10 lowest-confidence and the 10 misclassified strokes with the highest confidence, with player, level, and class.
- Per-player performance spread: which players are hardest and whether that relates to level.
- Plots of several failure cases next to typical strokes of the true and predicted class.
- Accept when: `docs/ERRORS.md` summarizes the patterns found, describing only what the data shows.

### D8: Inference interface and latency

- `racketimu.infer.infer(imu_window: np.ndarray, fs: float) -> InferenceResult` returning the stroke type, class probabilities, calibrated confidence, and predicted quality score. Add type hints and a docstring with an example.
- A CLI that runs inference on one player's file and writes JSON.
- Latency: median and p95 per window on the laptop CPU over at least 1000 runs, with the hardware reported. Raspberry Pi 4 latency is optional, if the owner sets one up.
- Accept when: the interface has a test and the CLI runs end to end.

### D9: Reporting

- `make report` generates `results/SUMMARY.md` from the saved results only. It includes the dataset card summary, every results table (D3 to D6), calibration, latency, the main error-analysis findings, and limitations.
- The limitations must include:
  - table tennis rather than tennis
  - forehands only
  - shadow strokes rather than ball strikes (confirmed by the authors' papers)
  - a very low observed sampling rate (about 9 rows per second, versus 100 Hz / 70 Hz stated)
  - coach-averaged scores only, so no coach-agreement ceiling
  - no synchronized video or motion capture, so body motion cannot be inferred or validated
  - 16 players
  - a single sensor placement
- Future work: collect synchronized racket-IMU and video or motion-capture recordings to extend this from stroke recognition to body-motion inference.
- Fill in the README results section with figures and the dataset citation at the top. Tag `v0.1` and push the tag.

## Commands

```
make test
make inspect     # D1
make preprocess  # D2
make classify    # D3
make quality     # D4
make skill       # D5
make ablation    # D6
make errors      # D7
make latency     # D8
make report      # D9
make all         # D2 to D9 in order
```

## Code style

Python 3.11, type hints, `ruff` clean, no notebooks in the core pipeline. Every random process takes its seed from the config. CPU is the default device for all reported results.
