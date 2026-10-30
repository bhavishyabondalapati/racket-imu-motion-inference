# CLAUDE.md: Racket IMU Motion Inference

## Project goal

Infer tennis movement from a single racket-mounted IMU (a smartphone taped to the racket handle), using synchronized video as the reference. The project answers one question: how much body and equipment movement can be inferred reliably from sparse inertial data, and where does it fail?

Three tasks, in order of priority:

1. Stroke detection: find individual strokes in a continuous IMU stream.
2. Stroke classification: forehand, backhand, serve (volley optional).
3. Motion inference: predict the hitting-arm wrist trajectory from the IMU window, evaluated against pose estimated from video.

Evaluation quality matters more than model complexity. The headline protocol is leave-one-player-out (LOPO) cross-validation, so every result reflects performance on a player the model has never seen.

Owner: Bhavishya Bondalapati. Package name: `racketimu`. Repo name: `racket-imu-motion-inference`.

## Ground rules (non-negotiable)

1. Never fabricate data or results. Every number in `results/`, the README, or any summary must come from a script run on real recordings in `data/raw/`. Synthetic data is allowed only for unit tests, lives only in `tests/`, and must never appear in results.
2. If something cannot be computed (too little data, a missing file), say so plainly in the output. Do not fill gaps with placeholder or estimated numbers.
3. Inspect before assuming. Before writing any loader, look at the actual files in `data/raw/` (headers, units, timestamp format, sampling rate) and adapt to what is there. Do not assume a specific app's export format.
4. Never commit raw data or video. `data/raw/`, `data/interim/`, and all `.mp4/.mov` files are git-ignored. Videos contain other people's faces.
5. Document every design decision in `docs/DESIGN.md` as you go (what was chosen, the alternatives, and why). The owner must be able to explain every choice in an interview.
6. Small, clear commits. Commit after each logical step. Push to `origin main` at the end of each milestone. Never force-push.
7. Ask the owner before: changing the evaluation protocol, deleting or overwriting data, adding a heavy dependency, or skipping an acceptance criterion.
8. Stop at the end of each milestone and summarize what was built, what the owner should check (including plots to look at), and any open questions.

## Recording protocol (done by the owner, not Claude Code)

- IMU: a smartphone taped firmly to the racket throat or handle with a fixed orientation. Record the accelerometer and gyroscope (and the magnetometer if available) at 100 Hz or higher with Sensor Logger or phyphox, then export CSV.
- Video: a second phone on a tripod, 60 fps, side-on view of the player, full body in frame.
- Sync: at the start and end of every recording, tap the racket strings sharply against the palm. That creates an IMU spike and an audible, visible event in the video. Two taps allow estimating both offset and clock drift.
- Calibration: at the start of each session, hold the racket still for 3 seconds lying flat, then 3 seconds standing vertically on its butt cap.
- Sessions: one stroke type per recording (e.g. 15 to 25 forehands), so the session label gives the stroke label.
- Players: at least 3, ideally 5. Record handedness and approximate level. Get each player's consent to be filmed.

## Repository structure

```
racket-imu-motion-inference/
  CLAUDE.md
  README.md
  pyproject.toml            # pinned dependencies
  Makefile                  # one command per pipeline step
  configs/default.yaml      # all thresholds, window sizes, seeds
  data/
    sessions.yaml           # metadata (committed)
    raw/                    # git-ignored
    interim/                # git-ignored
    processed/              # small derived tables (committed only if no personal data)
  src/racketimu/
    io.py                   # loaders for IMU CSV and video metadata
    sync.py                 # tap detection, offset + drift estimation
    frames.py               # phone-to-racket rotation, orientation filter
    segment.py              # stroke detection
    reference.py            # MediaPipe pose extraction + normalization
    dataset.py              # windows, labels, LOPO splits
    features.py             # handcrafted features for baselines
    models.py               # 1D CNN, GRU classifier/regressor
    train.py
    evaluate.py             # metrics, calibration, per-player breakdowns
    infer.py                # public inference interface
    replay.py               # overlay video generation
    report.py               # builds results/SUMMARY.md
  scripts/                  # thin CLI wrappers
  tests/                    # pytest, synthetic data only
  results/                  # one folder per run + SUMMARY.md
  docs/DESIGN.md
  docs/figures/
  .github/workflows/tests.yml
```

## Metadata format: `data/sessions.yaml`

```yaml
players:
  p01: {handedness: right, level: intermediate}
sessions:
  - id: p01_forehand_s1
    player: p01
    stroke: forehand
    imu_dir: data/raw/p01_forehand_s1/imu/
    video: data/raw/p01_forehand_s1/video.mp4
    sync:
      start_tap_video_time_s: null   # auto-detected; manual override allowed
      end_tap_video_time_s: null
    exclude_stroke_indices: []       # e.g. mishits the owner flags
    notes: ""
```

## Milestones

### M0: Scaffold

- `pyproject.toml` with pinned versions (numpy, pandas, scipy, scikit-learn, torch, opencv-python, mediapipe, ahrs, matplotlib, pyyaml, pytest, ruff).
- `.gitignore` covering data, videos, model checkpoints over 5 MB, and environment folders.
- Config loading, a seed utility, and a `Makefile` with `make test`, `make lint`, and placeholder targets for each step.
- A GitHub Actions workflow that runs `pytest` and `ruff` on push.
- README skeleton (goal, setup, how to reproduce; results section left empty).
- Accept when: `make test` passes in CI and the first push to GitHub succeeds.

### M1: Loading and synchronization

- Inspect real files first, then write loaders returning a DataFrame with a common time base in seconds, accelerometer in m/s², and gyroscope in rad/s (convert units if needed and document it).
- Resample IMU streams to one uniform rate (configurable, default 100 Hz) and report the original rates.
- Detect start and end taps in the IMU (acceleration-magnitude peaks within the first and last N seconds).
- Detect the taps in the video from the audio track (extract with ffmpeg, then onset/peak detection), with a manual override from `sessions.yaml`.
- Fit a linear mapping (offset + drift) between video time and IMU time from the two taps. Report the offset, the drift in ppm, and the residual.
- Save a verification plot per session to `docs/figures/sync/`.
- Accept when: every session has a sync result or a clear failure message, and the plots show aligned spikes.

### M2: Coordinate frames and orientation

- Estimate the phone-to-racket rotation from the two static calibration poses using the gravity direction. Store it per session.
- Express IMU data in the racket frame (define the axes in DESIGN.md: e.g. x along the handle, z normal to the string bed).
- Estimate orientation with a Madgwick filter (`ahrs`), using a complementary filter as a cross-check.
- Accept when: gravity in the static poses points along the expected racket axis within a reported angular error.

### M3: Stroke segmentation

- Detect strokes as peaks in gyroscope magnitude, with a threshold and a minimum spacing from the config. Exclude the sync taps.
- Extract fixed windows around each peak (default -0.6 s to +0.4 s, configurable).
- Write `data/processed/strokes.parquet` with stroke_id, session, player, label, t_peak, and quality flags.
- Report the detected count against the owner's expected count per session (add an `expected_strokes` field to sessions.yaml if useful).
- Accept when: plots of detected peaks look correct and the counts match the owner's notes.

### M4: Video reference

- Run MediaPipe Pose (use whichever API the installed version supports, and pin it) on each video. Keep the hitting-side wrist, elbow, and both shoulders, with visibility scores.
- Normalize keypoints: subtract the shoulder midpoint and divide by shoulder width.
- Map frames to IMU time with the M1 sync. Resample each stroke's reference wrist trajectory to K points (default 32).
- Flag low-visibility frames. Mark strokes above a configurable occlusion fraction as unusable for regression, and report how many were excluded and why.
- Estimate reference noise from jitter during the static calibration periods, and report it as a floor on achievable error.
- Accept when: a sample overlay of keypoints on video frames looks right (save a few frames to `docs/figures/reference/`; only frames of the owner).

### M5: Stroke classification

- Splits: outer LOPO, with an inner validation split by player inside the training players (for early stopping and temperature scaling). No stroke from the test player may influence training or calibration.
- Baselines: majority class, then a random forest on handcrafted features (per-axis statistics, peak angular velocity, orientation change, window energy).
- Deep models: a small 1D CNN and a small GRU in PyTorch. Run 3 seeds and report mean ± std.
- Metrics: accuracy, macro-F1, confusion matrix, and per-player and per-class results.
- Calibration: expected calibration error before and after temperature scaling, plus a reliability diagram.
- Accept when: results are saved under `results/<run_id>/` with the config, git commit hash, and metrics JSON.

### M6: Wrist trajectory inference

- Predict the K-point normalized 2D wrist trajectory from the IMU window.
- Baselines: the global mean trajectory, and a class template (the mean training trajectory of the predicted stroke class).
- Model: a GRU or CNN regressor under the same LOPO protocol.
- Metrics: mean per-point error in shoulder-width units, error at the impact point, and per-class and per-player breakdowns, compared against the reference noise floor from M4.
- Accept when: the report states clearly whether the model beats the class-template baseline, and by how much on unseen players.

### M6b: Sparsity ablation and embedded latency

- Re-run the best classifier and the trajectory model under LOPO with:
  - (a) sampling rate 100, 50, 25 Hz;
  - (b) gyroscope only, accelerometer only, both;
  - (c) window length at 100%, 75%, 50% of default.
- Report each condition against the full-data result and the class-template baseline, with mean ± std over seeds. Plot error vs sampling rate.
- Measure inference latency (median, p95) on a Raspberry Pi 4 in addition to the laptop CPU. Document the Pi setup in DESIGN.md.
- Accept when: SUMMARY.md has an ablation table and states which reductions hurt most.

### M7: Inference interface, latency, replay

- `racketimu.infer.infer(imu_window: np.ndarray, fs: float) -> InferenceResult`, where InferenceResult holds the stroke label, class probabilities, calibrated confidence, and the predicted wrist trajectory. Add type hints and a docstring with an example.
- A CLI: `python -m racketimu.infer --session <id>` that runs segmentation and inference on a full session and writes JSON.
- Latency: median and p95 per window on CPU over at least 1000 runs. Report the hardware.
- Replay: `scripts/make_replay.py` writes an MP4 showing the video with the reference and predicted wrist paths overlaid, plus the label and confidence. Replays are git-ignored.
- Accept when: the interface is covered by a test and the CLI runs end to end on one session.

### M8: Reporting

- `make report` generates `results/SUMMARY.md` from the saved result files only. It must include:
  - dataset statistics (players, sessions, strokes per class and per player, exclusions with reasons)
  - sync offsets, drift, and residuals
  - the classification table (all models, LOPO, mean ± std) with per-player results
  - calibration results
  - the regression table against both baselines and the noise floor
  - latency
  - the 5 lowest-confidence and 5 highest-error strokes, with session ids, for the owner to review
- Fill in the README results section from SUMMARY.md, with figures.
- Add a Limitations section to DESIGN.md based on the observed failure cases (single camera and 2D reference, small player count, amateur players, phone mass on the racket, and anything else the data shows).
- Tag the release `v0.1` and push the tag.

## Commands

```
make test        # unit tests (synthetic data)
make sync        # M1
make frames      # M2
make segment     # M3
make reference   # M4
make classify    # M5
make trajectory  # M6
make latency     # M7
make report      # M8
make all         # everything after test, in order
```

## Code style

Python 3.11, type hints, `ruff` clean, functions under about 50 lines where reasonable, no notebooks in the core pipeline (an exploration notebook is fine under `notebooks/`). Every random process takes its seed from the config.
