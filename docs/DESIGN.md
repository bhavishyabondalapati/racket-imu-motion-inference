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

### CI runner pinned to `ubuntu-26.04`; actions on current majors
- **Chosen:** `runs-on: ubuntu-26.04`, `actions/checkout@v7`, `actions/setup-python@v7` (the latest major versions on 2026-10-30).
- **Alternative:** `ubuntu-latest`, which GitHub moves to new Ubuntu releases on its own schedule.
- **Why:** if the label moved, the test environment would change without any commit in this repo, and a new failure (or a hidden behaviour change) couldn't be traced to a diff. 26.04 is the image the first green run already used, so pinning it changes nothing today. Upgrading is now a deliberate one-line change. The old v4/v5 actions also triggered a Node.js 20 deprecation warning.

## Switch to a public dataset (2026-10-31)

### Use Tabrizi et al. (2020) instead of self-recorded sessions
- **Chosen:** the public table tennis dataset by Tabrizi, Javani and Pashazadeh (Mendeley Data, DOI 10.17632/b7bc9y232m.3, CC BY 4.0): one BNO055 IMU on the racket face, 16 players (8 novice, 8 professional), 1570 forehand samples of three types, labelled and scored by 3 coaches.
- **Alternative:** the original plan, which used self-recorded tennis sessions with a phone on the racket and synchronized video as the reference (old milestones M1 to M8).
- **Why:** it has 16 players, including both novices and professionals, plus labels and quality scores from three coaches. A self-recorded pilot would have had 3 to 5 amateur players and no expert scores. That makes leave-one-player-out evaluation, skill-level transfer, and quality estimation possible now.
- **What is lost:** there is no synchronized video or motion capture, so the wrist-trajectory task (old M6) and anything about body motion can no longer be validated and is dropped. It is also table tennis, not tennis. Both go into the D9 limitations.
- **What is not claimed:** the data and its original analysis belong to the authors. This project's contribution is its evaluation protocol, analysis, ablations, and inference interface.

### Makefile targets follow the new milestones
- **Removed:** the recording-based placeholders `sync`, `frames`, `segment`, `reference`, and `trajectory` (video sync, phone-to-racket frames, pose reference, wrist trajectory).
- **Added:** placeholders `inspect` (D1), `preprocess` (D2), `classify` (D3), `quality` (D4), `skill` (D5), `ablation` (D6), `errors` (D7), `latency` (D8), and `report` (D9). As before, each one exits with an error until it is built. `make all` runs D2 to D9.
- **Kept for now (M0 code):** `make models` / `scripts/download_pose_model.py`, the `pose_model`, `sync`, `calibration`, `orientation`, and `reference` sections of `configs/default.yaml`, and ffmpeg in the env. None of these is used by the new plan. Removing them is left to the owner's decision, because the spec says M0 stays as is.

### Recording-plan code removed; recoverable from the `recording-plan-archive` tag
- **Removed:** `opencv-python`, `mediapipe`, and `ahrs` from `pyproject.toml`. Also the pose-model download (`scripts/download_pose_model.py`, `make models`), the `sync`, `calibration`, `orientation`, `reference`, and `pose_model` config sections, the `paths.sessions` entry, and `data/sessions.yaml`. `REQUIRED_SECTIONS` in `config.py` no longer lists `sync` or `reference`.
- **Where it lives:** the annotated tag `recording-plan-archive` (commit `47a9902`) points to the last commit that still has all of it. `git checkout recording-plan-archive` restores it, for example to build the video extension later. The M0 entries above about MediaPipe and ffmpeg describe that archived state.
- **Rotation math:** D2 will use `scipy.spatial.transform.Rotation` for Euler angles and frame changes instead of `ahrs`. scipy is already a dependency, it handles any Euler order explicitly (e.g. `"ZYX"` vs `"zyx"` for extrinsic vs intrinsic), and the dataset already supplies orientation from the BNO055, so no filter library is needed.
- **Kept:** `racketimu.assets` (SHA-256 helpers) is generic and tested, and can record checksums of the raw dataset files. ffmpeg stays in the local conda env only; nothing in the repo depends on it.
- **Why remove instead of leaving it:** unused dependencies slow CI installs (mediapipe and opencv are large), and dead code makes a reviewer wonder what is actually used.

## D1: Data inspection and loader (2026-10-31)

### Sample set: the authors' cleaned ids, rows from the raw files, minus unusable files
- **Chosen (owner-approved):** start from the 1570 ids in the authors' `Processed/` folders, load each one's rows from `Raw/Sensory-Data/<id>.csv`, and exclude the 3 empty raw files and the 223 files whose accelerometer and gyroscope are zero in every row, leaving 1344. The 157 ids the authors removed are not re-added.
- **Alternatives:** all 1728 raw files (re-adds samples the authors chose to drop); the processed files as they are (zero-padded, and 223 of them carry no motion data); the processed versions for the 19 ids where they differ from raw (see below).
- **Why:** the authors' cleaned set matches their published description (1570 samples, 16 players), so starting there respects their curation. Raw files hold the same rows without the padding, which keeps padding a choice made in D2. Files with no accelerometer or gyroscope signal cannot inform a model of racket motion.
- **Finding kept in view:** for 19 ids (1223, 1300-1312, 1314-1318, player N02) the processed file contains different rows from the raw file: non-zero accelerometer and gyroscope data and different timestamps. Under the approved rule these are excluded as zero-sensor files. Using the processed versions would add 18 N02 pushes and 1 N02 basic, but it is an open question for the owner, because their origin is not documented.
- **Every exclusion** is listed with its reason and id ranges in `docs/DATASET.md`, generated by `make inspect`.

### Exclusion rules in priority order, one reason per sample
- **Chosen:** each sample gets the first rule that excludes it: removed by the authors, then in the clean CSV only, then empty raw file, then all-zero acc and gyro.
- **Why:** one reason per sample makes the counts add up exactly (1728 = 1344 + 157 + 1 + 3 + 223) and keeps the reconciliation table honest.

### Sample 1313 is not in the authors' set
- It appears in `Clean-Samples-Scores.csv` (1571 rows) but has no processed file and is missing from `Clean-Players-Level.xlsx`. Two of the authors' three sources leave it out, and leaving it out gives their stated 1570.

### Sample 720 belongs to P05 (owner-approved)
- The scores CSV says P06; both spreadsheets put 720 at the end of P05's push range (651-720). Contiguous id blocks per player and stroke are the pattern across the whole dataset, so the spreadsheet is followed. The override lives in `configs/default.yaml` (`dataset.player_overrides`) and is reported in `docs/DATASET.md`.
- **Consequence:** P06's only usable sample was 720, so P06 has no usable acc/gyro samples.

### 15 LOPO folds for acc/gyro models; pooled out-of-fold metrics (owner-approved)
- **Chosen:** P06 is never a test player for models that use acc/gyro (`lopo.excluded_test_players`). Overall confusion matrix and macro-F1 come from pooling all out-of-fold predictions; per-fold results are reported alongside.
- **Alternatives:** 16 folds with an empty P06 fold; averaging per-fold macro-F1.
- **Why:** a fold with no test samples has no meaning. Averaging per-fold macro-F1 is unstable here because several players have only one or two classes (P01 basics only, N02 basics only once exclusions apply), which makes a per-fold macro-F1 undefined or misleading. Pooling first gives one well-defined number, and per-fold results still show the spread across players.

### Units from BNO055 defaults, checked against the data
- **Chosen:** acc raw/100 → m/s², gyro raw/16 → deg/s → rad/s, mag raw/16 → µT, Euler angles kept in degrees. Converted in `racketimu.io.to_physical`; scales in `dataset.units`.
- **Evidence:** the median acceleration magnitude at rest is 981 raw (9.81 m/s²); the rotation rate implied by consecutive Euler angles gives 16-18 raw per deg/s; the magnetometer's median magnitude is 663 raw (41 µT, within Earth's field range).
- **Why rad/s for gyro:** SI units throughout. Euler angles stay in degrees because they are only used as input channels, not in rotation math.

### Sensor frame only; no world-frame or gravity-removal transform (owner-approved)
- **Chosen:** inputs will be sensor-frame acceleration and angular velocity plus the Euler angles as separate channels.
- **Alternative:** rotate acceleration into a gravity-aligned frame using the Euler angles and subtract gravity (the original D2 plan).
- **Why:** the Euler angle order (and whether angles are intrinsic or extrinsic) is not documented, and a test of six candidate orders against the gyroscope could not tell them apart at this sampling rate. A transform built on a guessed order could quietly corrupt every input. Feeding the Euler angles as channels lets a model use the orientation without assuming the convention.

### Keep duplicate rows, report the rate
- 21% of consecutive row pairs are exact repeats (most likely the logger reading faster than the sensor updates, but that is not documented). They are kept so that row positions stay as logged; the rate is in `docs/DATASET.md`.

### Timestamps are decoded, but only used for reporting
- `Time/Date` cells are the Python repr of GBK-encoded Chinese date text that was mis-decoded as Latin-1 (e.g. `2019年9月17日10时43分20秒`). `decode_timestamp` reverses this. They have 1-second resolution (91 used files stop at minutes), so they are used only to measure rows per whole second, never as a time axis for models.

### Fixed-length windows by row count, not by time (owner-approved; built in D2)
- **Chosen:** each stroke is padded or cropped to 32 rows centered on peak gyro magnitude, with a mask (`window` in the config). The 100 Hz resampling setting is removed. The D6 rate ablation keeps every row, every 2nd row, and every 4th row (`ablation.row_strides`).
- **Why:** at about 8 rows per second with 1-second timestamps, a time-based window (the old 0.6 s before / 0.4 s after the peak) would be about 8 rows and could not be placed precisely. Resampling to a uniform rate is not possible without per-row times.

### Stated vs observed sampling rate reported side by side
- The authors state 100 Hz (Data in Brief, 2020) and 70 Hz (Computational Intelligence and Neuroscience, 2021); the files show about 8 rows per whole second. All three are reported in `docs/DATASET.md` without speculating about the cause. Likewise the 2021 paper's 1525 samples per criterion versus 1570 in the cleaned set.

### No coach-agreement ceiling
- C1-C5 are five criteria, each the average of 3 coaches; individual coach scores are not published. Agreement between coaches cannot be computed, so D4 has no agreement ceiling. Correlations between criteria are reported instead and labelled clearly as such.

### `make inspect` writes the dataset card from code
- `scripts/inspect_dataset.py` → `racketimu.datacard.build_dataset_card`. Every number in `docs/DATASET.md` is computed from the files on each run. When a number cannot be derived, the card says "not computable" instead of failing or guessing. The exploratory scripts used for the first inspection report were not committed; their findings are all reproduced by this code.

## D2: Preprocessing (2026-10-31)

### Heading channels kept out of the primary input (owner's decision)
- **Chosen:** primary input = acc x/y/z, gyro x/y/z, roll, pitch (as sin/cos, see below): 9 channels (`channels.primary`). Yaw and the magnetometer (`channels.heading`) are used only in a D3 with/without check and a D6 "+ heading" condition.
- **Why:** yaw and the magnetometer measure absolute heading relative to the room. Each stroke type was recorded in its own block, so if the player faced a slightly different direction per block, heading alone could identify the class. That would be a shortcut that says nothing about stroke shape and would not transfer to a new room or player. D3 also runs a random forest on per-sample channel means only (no motion shape) to test for player or session leakage.

### Pitch (and yaw) as sin/cos
- **Finding:** pitch jumps by more than 180° between consecutive rows in 163 of 1344 used strokes, and yaw in 145. These are wrap-arounds at ±180° (visible in `docs/figures/d2/examples_roll_pitch.png`, ids 77 and 283), not real motion. Roll stays within ±90° and never wraps.
- **Chosen:** add `pitch_sin`, `pitch_cos`, `yaw_sin`, `yaw_cos`; the primary input uses the pitch pair instead of raw pitch degrees. Raw degrees stay in `strokes.parquet`.
- **Alternatives:** raw degrees (a model would see a fake 358° jump); unwrapping within each stroke (continuous, but the same orientation can end up as 170° in one stroke and 530° in another).
- **Why:** sin/cos is continuous everywhere and gives the same values for the same orientation in every stroke. The cost is one extra channel.

### Window: 48 rows, peak gyro row at position 16, zero padding with a mask
- `racketimu.dataset.window_stroke` puts the first row of maximum gyro magnitude at `window.peak_position` (`pos` 0) and copies the rows around it. Rows outside the stroke are zero with `mask = False`. No scaling here: scaling is fitted inside each training fold in D3 to avoid leakage.
- **First version (approved in review):** 32 rows centered on the peak kept only 77.4% of real rows; 664 of 1344 strokes lost rows, mostly after the peak, because the peak tends to sit early in a stroke (median at 29% of its length).
- **Comparison (owner's request): 48 rows, peak at three positions, scored on coverage only.** Coverage is the fraction of real rows kept. It uses no labels, so choosing on it cannot leak class information into the design.

  | peak position (rows before / after) | real rows kept | strokes losing rows |
  |---|---|---|
  | 24 (24 / 23) | 30844 of 34538 = 89.30% | 263 |
  | **16 (16 / 31)** | **31137 of 34538 = 90.15%** | **223** |
  | 12 (12 / 35) | 30452 of 34538 = 88.17% | 316 |

- **Chosen:** 48 rows with the peak at position 16 (`window.n_rows: 48`, `window.peak_position: 16`), the highest coverage. It matches the shape of the data: few rows before the peak, more after. The D2 figures were regenerated with this window.

### Finding: the logging rate probably undersamples the swing (interpretation, recorded for D3 and the limitations)
- **Numbers** (used set, also in `docs/DATASET.md` under "Signal characteristics"): the peak rotation rate per stroke has a median of 1.22 rad/s (70 deg/s), 95th percentile 3.0 rad/s. The acceleration magnitude has a median of 9.81 m/s² and a 1st-99th percentile range of 8.79-11.44 m/s² over all rows, where gravity alone is 9.81. Rows arrive at about 8 per second.
- **Interpretation, supported by the numbers but not certain:** the logging rate undersamples the fast phase of the stroke, so the signal mostly reflects the racket's orientation rather than swing dynamics. This matters for what the models can learn: stroke type may be recognisable from how the racket is held and turned, while fine timing and speed of the swing are largely invisible.
- **The peak is on the first row in 238 strokes (17.7%):** the file starts at its largest rotation rate, so those strokes have no rows before the peak. This is a likely sign that recordings sometimes start mid-swing. The share is similar across classes (Basic 15.6%, Push 21.4%, Topspin 17.9%) and levels (novice 17.3%, pro 18.0%).
- **Class profiles overlap heavily** (`class_profiles.png`): topspin has the highest median peak rotation rate and push the lowest, with wide interquartile bands.

### `strokes.parquet` is not committed
- **Chosen:** git-ignore `data/processed/*.parquet`; `make preprocess` rebuilds it in seconds.
- **Why:** it is a re-packaging of the authors' rows. Keeping it out keeps the README's statement true that the data is not redistributed here (reusers download it from Mendeley Data), and avoids a stale copy drifting from the code. It contains no personal data (anonymous player codes only), so committing it would also be acceptable under the spec.
- **Format:** long table, one row per (sample, window position): metadata, `pos` (0 = peak row), `mask`, all channels (raw Euler degrees plus sin/cos), `peak_row`, `rows_cropped`. `racketimu.dataset.to_arrays` turns it into `X [N, 32, C]`, `mask [N, 32]`.

### Review figures (`docs/figures/d2/`)
- 3 example strokes per class (seeded choice from different players, mixing levels) for gyro, acc, and roll/pitch, with padding shaded; per-class median and interquartile band of |gyro| and |acc|; window coverage per position.
- Colors are the first three categorical slots of the reference palette, validated for colour-vision deficiency. Each series also has its own line style and a legend, so identity never relies on color alone (the third color is low-contrast on white).

## D3: Stroke type classification (2026-10-31)

### Protocol
- **Outer loop:** LOPO over the 15 players with usable samples (P06 has none). The test player's samples are only ever used for prediction.
- **Inner validation:** in each fold, 1 novice and 1 pro are taken out of the training players as a validation set (`classify.inner_val_players`). Players with all three classes are preferred, so early stopping and temperature scaling see every class. The pick depends only on `seed + fold index`, so every model and seed in a fold uses the same split, which keeps comparisons fair.
- **Same split for every model:** random forests are also fitted on the fit set only (training players minus validation players), and their temperature is fitted on the validation players. This costs the forest some training data, but every model is then trained, calibrated, and tested the same way.
- **No leakage:** handcrafted features are computed per sample; scaling (deep models) is fitted on fit-set real rows only and padded rows are reset to zero afterwards; hyperparameters are fixed in the config, not tuned on any fold.

### Class weighting
- **Chosen:** `class_weight: balanced` for the forests and the same inverse-frequency weights in the deep models' cross-entropy, computed on each fold's fit set.
- **Why:** the headline metric is macro-F1 and the classes are imbalanced (672 basic, 364 push, 308 topspin). Without weights, models drift toward predicting basic. The majority baseline is left unweighted, as the reference for "no learning".

### Features for the random forests (`racketimu.features`)
- Per channel over real rows: mean, std, min, max. Motion features: peak, mean, and energy of gyro magnitude; mean, std, and peak of acceleration magnitude and its energy around gravity; roll and pitch change from first to last real row; total roll and pitch path. Pitch differences are wrapped, so a crossing of ±180° counts as a small change.
- **Deliberately not included:** the number of real rows. Stroke file length differs by class (basic files are shorter; see `docs/figures/d2/window_coverage.png`), and length reflects how the files were cut as much as the motion. It is tested separately in the length check instead.

### Models
- **1D CNN:** two convolutions (32 and 64 channels, kernel 5), ReLU, masked mean over real rows, dropout 0.2, linear head. **GRU:** one layer (64 hidden units), masked mean over real rows, dropout 0.2, linear head. Both get the mask as an extra input channel.
- **Training:** Adam (learning rate 0.001), batch 32, up to 200 epochs, early stopping on validation loss with patience 20, best epoch restored. CPU, 3 seeds.
- **Why so small:** about 1000 training samples per fold. Larger models would mostly memorise players.

### Calibration
- Temperature scaling, fitted per fold on the inner validation players by minimising negative log-likelihood. For the forests it is applied to log-probabilities (probabilities clipped at 1e-6), so the same method works for every model. Temperature never changes the predicted class, only the confidence.
- ECE uses 15 equal-width bins on the top-class confidence, over pooled out-of-fold predictions.

### Metrics
- Overall accuracy, macro-F1, and the confusion matrix come from pooled out-of-fold predictions (owner-approved), reported as mean ± std over 3 seeds.
- Per test player, macro-F1 is averaged over **the classes that player has**, because 5 of the 15 test players have only basic strokes or only two classes. These per-player numbers show spread; they are not comparable to the pooled number.

### Shortcut checks, with decision rules fixed before the run
- **Heading check:** the handcrafted forest with and without yaw + magnetometer features. A pooled macro-F1 gain of at least 0.02 counts as "noticeable" and is flagged as a likely shortcut.
- **Leak check:** a forest on the per-sample mean of each primary channel only (no motion shape). A macro-F1 at least 0.10 above the majority baseline counts as "well above". **Caveat fixed in advance:** channel means also encode the racket's average orientation, which can be a genuine difference between stroke types. A high score is a warning sign, not proof of leakage.
- **Length check (added after D2):** a forest on the real-row count only. Same 0.10 rule.
- Thresholds live in `classify.checks` and were committed before the first run.

### "Best model" for later milestones
- Choosing the best model by its LOPO test score, as D4 and D6 ask, is a small optimistic bias, because the selection uses the test folds. It is acceptable for picking which model to study further, but the chosen model's score should not be read as a clean estimate. This is noted wherever the best model is used.

### Results layout (`results/<run_id>/`)
- `report.md` (all tables), `metrics.json` (per-seed metrics, checks, confusion counts), `oof_predictions.csv.gz` (every out-of-fold prediction with raw and calibrated probabilities), `config.yaml`, `run.json` (git commit, dirty flag, hardware, versions, duration), `confusion.png`, `reliability.png`. These are committed: they are small, contain no raw data, and the README must cite them.

### D3 results and findings (run `d3_20261101T032754Z`, commit `ab928dd`)
All numbers below are from `results/d3_20261101T032754Z/report.md`: pooled out-of-fold over 15 LOPO folds, mean ± std over 3 seeds, uncalibrated.

| model | macro-F1 | accuracy |
|---|---|---|
| Majority class | 0.222 | 0.500 |
| RF, handcrafted (primary) | 0.798 ± 0.007 | 0.808 ± 0.006 |
| 1D CNN | 0.786 ± 0.016 | 0.794 ± 0.008 |
| GRU | 0.788 ± 0.014 | 0.792 ± 0.017 |
| Leak check: RF on channel means only | 0.803 ± 0.003 | 0.813 ± 0.003 |
| Motion-only: RF with channel means removed | 0.667 ± 0.006 | 0.697 ± 0.005 |
| Heading check: RF + yaw and magnetometer | 0.802 ± 0.005 | 0.814 ± 0.006 |
| Length check: RF on real-row count | 0.307 ± 0.003 | 0.314 ± 0.004 |

- **Reproduction check (owner's request):** the first run and an identical rerun gave the same predicted class for every one of the 25,536 out-of-fold rows, and bit-identical probabilities for the majority baseline and every random forest. The CNN and GRU differed in probabilities (up to 0.11) for one fold only (N01, seed 0). **Cause:** `run_deep` built the model before seeding, so the very first deep model's initial weights depended on whatever random state the process had. Fixed in `ab928dd` (seed set before building the model, with a unit test). In a fresh process with a deliberately different prior random state, the fixed code reproduces the saved N01 / seed 0 probabilities exactly (max difference 0.0) for both models. The deep-model numbers above come from the fixed code; the two earlier runs are superseded and not kept.
- **Calibration:** uncalibrated probabilities are the primary result (owner's decision). Temperature scaling fitted on the 2 inner validation players **did not transfer** to the unseen test player: ECE got worse for every model (RF 0.052 → 0.089, CNN 0.070 → 0.108, GRU 0.105 → 0.122, mean-only RF 0.048 → 0.082). For the length check the fitted temperature hit the upper bound (20), which only flattens a model that has learned almost nothing. Two players are too few, and too unlike the test player, to estimate a temperature that generalises.
- **Leak-check interpretation (hedged):** a forest that sees only each sample's average channel values (mean acceleration direction, mean gyro, mean roll and pitch) matches the full handcrafted forest (0.803 vs 0.798) and both deep models. Mean racket orientation alone carries essentially all the information the models use, so classification relies on **static racket angle rather than swing dynamics**. This is consistent with racket-face angle differences between push and drive strokes: push is recognised almost perfectly (99% recall), while basic and topspin, both drives, are the main confusion. It is also consistent with the D2 undersampling finding. **A recording-block artifact cannot be ruled out:** each stroke type was recorded in its own block, so a sensor-mounting or posture difference shared by most players' blocks would look the same. LOPO does remove artifacts specific to one player.
- **Motion-only check:** with each sample's channel means removed, the forest still reaches 0.667 macro-F1 (+0.445 over majority, −0.131 vs the full forest). So there is some class information in the motion itself, but much less than in the static orientation.
- **Heading check:** +0.004 macro-F1, below the pre-registered 0.02 rule, so not flagged. Yaw and the magnetometer add nothing measurable, and 6 of 15 test players get better with them while 9 do not.
- **Length check:** +0.085 over majority, below the pre-registered 0.10 rule, so not flagged, but close. File length carries a little class information on its own.
- **Error pattern:** push is almost never confused (99% recall for every model). Basic and topspin are confused in both directions (18-24% of basics predicted as topspin, 33-37% of topspins as basic).
- **Per-player spread is large:** per-player macro-F1 (over that player's own classes) for the RF ranges from 0.556 (P03) to 1.0 (N01, P01). N03, N04 and P04 are hard for every model.
- **Deep vs forest:** the CNN and GRU do not beat the handcrafted forest. Their seed std (0.014-0.016) is larger than the gaps between the three.
- **Best deep model for D4-D6:** GRU (0.788 vs CNN 0.786), chosen by the rule "highest mean pooled macro-F1". The two are tied within seed noise; see `docs/REVIEW_QUEUE.md`. Choosing on test folds is a small optimistic bias, as noted above.

## D6: Sparsity ablation (2026-11-01)

### Design
- Same protocol as D3: the same 15 LOPO folds, inner validation players, model hyperparameters, CPU and 3 seeds. Only the input changes. The RF runs on every condition with handcrafted features of the channels in that condition. Motion features are computed only for the channel groups present, so the D3 features are unchanged (verified bit-for-bit). The GRU, the best D3 deep model, runs on full data, every 4th row, and motion-only (owner's instruction, to save time).
- **Row stride:** keep window rows whose position relative to the peak is a multiple of k. The peak row is always kept, and the window spans the same time with k times fewer rows (about 8, 4, and 2 rows per second).
- **Window length:** 75% and 50% windows keep the peak at the same relative position (36 rows with 12 before the peak; 24 rows with 8 before).
- **Motion-only:** each sample's per-channel mean over real rows is removed, both from the RF features (as in D3) and from the GRU input.
- Δ is the per-seed difference from the same model on full data, then averaged.

### Results (run `d6_20261101T034227Z`, commit `8ab3698`)
| condition | RF macro-F1 | Δ vs full | GRU macro-F1 | Δ vs full |
|---|---|---|---|---|
| Full data (primary, every row, 48 rows) | 0.798 ± 0.007 | - | 0.788 ± 0.014 | - |
| Every 2nd row (~4 rows/s) | 0.808 ± 0.003 | +0.010 | | |
| Every 4th row (~2 rows/s) | 0.799 ± 0.004 | +0.002 | 0.797 ± 0.004 | +0.009 |
| Accelerometer only | 0.798 ± 0.003 | +0.000 | | |
| Gyroscope only | 0.735 ± 0.001 | −0.063 | | |
| Accelerometer + gyroscope | 0.801 ± 0.001 | +0.003 | | |
| Primary + heading | 0.802 ± 0.005 | +0.004 | | |
| Window 75% | 0.802 ± 0.004 | +0.004 | | |
| Window 50% | 0.808 ± 0.002 | +0.011 | | |
| Motion-only | 0.667 ± 0.006 | −0.131 | 0.629 ± 0.021 | −0.159 |

- **Reproduction check:** the full-data condition gives exactly the same out-of-fold predictions as D3 for both the RF and the GRU (all 4032 seed-sample pairs each).
- **What hurts most, stated plainly:** only two reductions hurt. Removing each sample's channel means (motion-only) costs 0.131 (RF) and 0.159 (GRU) macro-F1. Using the gyroscope alone costs 0.063. **Nothing else hurts:** cutting the rate to every 4th row (about 2 rows per second), halving the window, or dropping the gyroscope and the Euler angles (accelerometer only) all stay within about ±0.01 of full data, which is the size of the seed noise. Some reductions even score slightly higher, within noise.
- **Interpretation (hedged, consistent with D3):** the accelerometer at rest measures the direction of gravity, which is the racket's orientation, and orientation is what the models use. Anything that keeps orientation keeps the performance; the two conditions that remove it (motion-only, gyroscope only) lose it. For this dataset the answer to "how much sensor data is needed" is: very little, a 3-axis accelerometer at about 2 rows per second over the stroke. But that is because the task is solved through static racket angle, not swing dynamics, so it may not hold for tasks that need motion.

## D4: Stroke quality estimation (2026-11-01)

### Design
- **Targets:** C1-C5 and their mean, each predicted separately. The RF uses one regressor per target. The GRU has one output per target, trained jointly with MSE on fit-set-standardized targets, so it is still a separate prediction per criterion from one network (see `docs/REVIEW_QUEUE.md`).
- **Baselines:** the global training mean, the per-class training mean, and the level-only baseline (the training mean for the test player's level). Baselines use all training players (fit + validation), which makes them as strong as possible. The per-class baseline uses the test stroke's **true** class, which flatters the baseline, the conservative direction.
- **Models:** the RF regressor on the D3 handcrafted primary features (fit set only) and the GRU (fit set, early stopping on the validation players), same folds and seeds as D3.
- **Rule fixed before the run:** a model judges stroke quality beyond level only if its pooled MAE is below the level-only baseline's **in every seed** (`quality.beat_rule`).

### Results (run `d4_20261101T035101Z`, commit `613650d`)
| score_mean | MAE | Spearman |
|---|---|---|
| Global training mean | 12.24 | −0.829 |
| Per-class training mean | 11.70 | −0.137 |
| **Level-only baseline** | **4.98** | **0.574** |
| RF regressor | 10.85 ± 0.02 | 0.281 ± 0.002 |
| GRU | 9.90 ± 0.26 | 0.297 ± 0.023 |

- **Verdict, stated plainly: the models do not judge stroke quality beyond level.** Neither the RF nor the GRU beats the level-only baseline on any of the 6 targets. On the mean score the level-only baseline's error (4.98 points) is about half the models' (9.9-10.9). The models do beat the global and per-class means slightly, mostly because they partly recover level from the signal.
- **Within level:** for novices, the models rank strokes weakly (Spearman 0.13 RF, 0.17 GRU on the mean score; about 0.35 on C3 for both). For pros, the rankings are negative (−0.15 RF, −0.23 GRU). Pro scores are tightly packed (std about 2 points on the mean), so there is little to rank.
- **Reading the baselines' Spearman:** the global-mean and level-only baselines show strongly negative Spearman (−0.83 overall; −0.72 within novices, −0.59 within pros). This is a known leave-one-out artifact: removing a low-scoring player from training raises the training mean predicted for that player, so predicted and true scores move in opposite directions across folds. It does not mean those baselines are anti-informative within a fold. Their MAE is the meaningful number.
- **Why, in the light of D3 and D6:** the signal mostly carries static racket orientation, which identifies stroke type well but says little about how well a stroke was played. Coach scores mostly track the player's level (novice mean 70.6, pro mean 93.7; `docs/DATASET.md`).

## D5: Skill level and cross-group generalization (2026-11-01)

### Design
- **Part 1, novice vs pro:** majority, RF (handcrafted primary features, balanced class weights) and GRU (2 classes, D3 recipe) on the D3 folds. Each test player has one level, so the results are pooled out-of-fold accuracy, balanced accuracy and macro-F1, plus per-player accuracy (owner's instruction), not per-fold F1.
- **Part 2, cross-group:** the stroke-type RF and the quality RF (one regressor per target), trained on one level and tested on the other. The reference is LOPO **within** the same level. Validation players (2) always come from the training group, chosen with the same preference for players with all three classes. As in D3/D4, the forests are fitted on the fit set only, so within-group and cross-group are trained the same way.

### Results (run `d5_20261101T035811Z`, commit `b89ac1f`)
| novice vs pro | accuracy | balanced accuracy | macro-F1 |
|---|---|---|---|
| Majority | 0.479 | 0.406 | 0.324 |
| RF | 0.775 ± 0.002 | 0.780 ± 0.002 | 0.772 ± 0.002 |
| GRU | 0.751 ± 0.023 | 0.754 ± 0.030 | 0.747 ± 0.026 |

| setting | stroke-type macro-F1 | score_mean MAE | score_mean Spearman |
|---|---|---|---|
| novices → novices (within-group LOPO) | 0.690 ± 0.013 | 10.27 | 0.063 |
| pros → novices (cross-group) | 0.767 ± 0.003 | 22.78 | 0.141 |
| pros → pros (within-group LOPO) | 0.839 ± 0.007 | 2.75 | −0.339 |
| novices → pros (cross-group) | 0.809 ± 0.014 | 25.09 | −0.071 |

- **Level can be predicted from the signal** (RF macro-F1 0.772 vs majority 0.324), but per-player accuracy varies from 0.51 (N06) to 1.0 (N02), and 0.56 for P08. **Major caveat, found in this milestone:** every novice was recorded on 2019-09-25 and every pro on 2019-09-17, -19 or -22 (`docs/DATASET.md`, "Recording dates by player"). Level is fully confounded with recording session, so part of the level signal may come from session differences (sensor mounting or calibration on that day) rather than skill. LOPO does not remove this, because the held-out novice shares the recording day with the training novices. This dataset cannot separate the two explanations.
- **Stroke type transfers across levels.** Training on pros and testing on novices gives a *higher* macro-F1 than training on other novices (+0.076). The pro training set is larger and has more push and topspin examples (novices are mostly basic strokes). Training on novices and testing on pros loses only 0.030 against within-pro LOPO. Nothing here suggests a stroke-type model built on one population fails on the other.
- **Quality does not transfer, as expected from D4.** The quality RF predicts scores typical of its training level: MAE rises by 12.5 points (testing on novices) and 22.4 points (testing on pros). Because scores mostly track level (D4), a quality model trained on one population would be badly miscalibrated on another, which matters for a consumer product whose users are mostly amateurs.
