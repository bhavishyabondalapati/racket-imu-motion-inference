# Morning summary (overnight run, 2026-11-01)

## What's done

All remaining milestones ran in the order you asked: **D3 → D6 → D4 → D5 → D7 → D8 → D9.** Each was committed and pushed, and CI was green before the next started. Every result folder was produced from a clean commit, recorded in its `run.json`. The protocol (15-fold LOPO, P06 not a test player, no leakage, CPU, 3 seeds) and the pre-registered thresholds were not changed. The repo is still private and there is no `v0.1` tag. No new heavy dependencies were added; `joblib` ships with scikit-learn.

Full numbers: [`results/SUMMARY.md`](../results/SUMMARY.md). Decisions with reasons: [`docs/DESIGN.md`](DESIGN.md). Everything for you to check: [`docs/REVIEW_QUEUE.md`](REVIEW_QUEUE.md).

## Headline numbers

| milestone | run (commit) | headline |
|---|---|---|
| D3 stroke type | `d3_20261101T032754Z` (`ab928dd`) | Macro-F1: RF **0.798 ± 0.007**, CNN 0.786 ± 0.016, GRU 0.788 ± 0.014, majority 0.222. Push 99% recall; errors are basic/topspin. |
| D3 checks | same | **Leak check flagged:** channel means alone reach 0.803. Motion-only RF 0.667. Heading +0.004 (not flagged). Length +0.085 over majority (not flagged, close). Temperature scaling made ECE worse for every model, so results are uncalibrated. |
| D6 ablation | `d6_20261101T034227Z` (`8ab3698`) | Only motion-only (−0.131 RF, −0.159 GRU) and gyroscope-only (−0.063) hurt. Every 4th row (~2 rows/s), half window, and accelerometer only all stay within ±0.011 of full data. |
| D4 quality | `d4_20261101T035101Z` (`613650d`) | **Does not beat the level-only baseline on any of 6 targets.** Mean-score MAE: level-only 4.98, RF 10.85, GRU 9.90. Weak within-novice ranking (Spearman 0.13-0.17); negative within pros. |
| D5 skill | `d5_20261101T035811Z` (`b89ac1f`) | Novice vs pro: RF macro-F1 0.772, GRU 0.747 (majority 0.324), **but level is fully confounded with recording day**. Stroke type transfers (pros→novices +0.076 vs within-novice LOPO; novices→pros −0.030); quality does not (MAE +12.5 / +22.4 points). |
| D7 errors | `docs/ERRORS.md` | 257/1344 errors, 237 of them basic/topspin. The 10 most confident errors come from N03/N04/N05 and follow the wrong class's racket angle. Error rate rises with stroke length (9% → 38%). Per-player accuracy 0.40 (N08) to 1.0. |
| D8 interface | `d8_20261101T040445Z` (`33ecc3a`) | `infer()` + CLI work end to end. Median **40.25 ms**, p95 40.88 ms per stroke (Apple M4 Pro, CPU, single thread, 1000 runs). |
| D9 report | `results/SUMMARY.md`, README | Generated from saved results only; README results, figures, and limitations filled in. No tag. |

## Things that did not reproduce, or needed fixing

1. **D3 deep models, first run vs rerun:** every predicted class was identical, but CNN/GRU probabilities differed (up to 0.11) for one fold (N01, seed 0). **Cause:** the model was built before seeding, so the first deep model's starting weights were unseeded. **Fixed** (seed before building, plus a unit test) and D3 re-run. Afterwards, the formerly broken case reproduces bit-for-bit in a fresh process. D6's full-data condition reproduces D3's RF and GRU predictions exactly (4032/4032 each), and a refactor of the training loop was verified not to change any output.
2. **D6 label bug:** the committed D6 run's own figures and report say "Every 2th row". The numbers are unaffected. `make report` redraws the D6 figures from the saved metrics with correct labels, and with a 0.5-0.9 y-axis so seed noise doesn't look like a trend.
3. Everything else ran once, cleanly, as planned.

## New findings you should know about

- **Level is confounded with recording day:** all novices were recorded on 2019-09-25, all pros on 2019-09-17/19/22. This is now a table in `docs/DATASET.md` and is listed in the limitations. It affects how D5's skill result (and the strength of D4's level baseline) can be read.
- **The models recognise stroke type from static racket angle, not swing dynamics** (D3 leak check, D6, D7 failure plots). Consistent with racket-face differences between push and drives and with the undersampling finding; a recording-block artifact cannot be ruled out.
- **Quality prediction is a negative result** under the rule you set.

## Review queue (from `docs/REVIEW_QUEUE.md`)

Decisions taken (the more conservative option each time):

1. Seeding bug fixed and D3 re-run; earlier runs superseded.
2. Best deep model = GRU (0.788 vs CNN 0.786, tied within noise), by the "highest mean macro-F1" rule.
3. Motion-only features: centered channels plus angle changes from the uncentered angles (exactly invariant to a per-sample offset).
4. Uncalibrated probabilities as primary (your decision).
5. D6 row stride keeps rows at multiples of k from the peak (peak always kept).
6. D6 shorter windows keep the peak at the same relative position.
7. D6 GRU only on full, every 4th row, and motion-only (your instruction).
8. D6 committed run has the "2th" label bug (numbers unaffected; summary figures redrawn).
9. D4 GRU predicts the 6 targets as 6 outputs of one network; the RF uses 6 separate models.
10. D4 per-class baseline uses the true stroke class (a stronger baseline).
11. D4 baselines use all training players; models use the fit set (both conservative).
12. D5 cross-group forests use the fit set only, like-for-like with within-group LOPO.
13. Recording-date table added to the D1 dataset card.
14. D8 confidence returned uncalibrated and flagged as such.
15. D8 keeps the quality estimate, with a note that it mostly reflects level.
16. D8 deployable models are the random forests.
17. D8 latency re-run so `run.json` points to a clean commit.

Open questions:

1. Calibration across players: try a temperature fitted on more players (an inner-protocol change, so not done)?
2. Leak check flagged: real racket-angle difference vs recording-block artifact. Not separable with this data.
3. Length check close to its threshold (+0.085 vs 0.10).
4. How to frame D6 (very little data is needed, but because the task is solved through static angle).
5. Keep or drop the quality output in the D8 interface, given D4's negative result.
6. Level confounded with recording day: carry this caveat on any skill claim.

## Not done (by instruction)

- No `v0.1` tag; the repo stays private; no Raspberry Pi latency.
