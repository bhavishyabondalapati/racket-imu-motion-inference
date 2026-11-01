# Review queue

## Owner decisions after review (2026-11-01)
- **Open question 1 (calibration across players):** do not refit on more players. The finding stays as written: temperature fitted on 2 inner validation players did not transfer, and confidences are reported uncalibrated.
- **Open question 5 (quality output):** removed from the default `infer()` output and CLI. The D4 results and code stay. Quality is available only behind an explicit flag (`experimental_quality=True`, CLI `--experimental-quality`), documented as "does not beat a level-only baseline; experimental". This supersedes decision 15 below.

Decisions taken while the owner was offline (2026-11-01), each with the more conservative option, plus open questions. Nothing here changes the evaluation protocol (15-fold LOPO, P06 not a test player, no leakage, CPU, 3 seeds) or the pre-registered check thresholds.

## Decisions

### D3
1. **Seeding bug fixed before the final run.** The reproduction check showed that the first deep model in a process got unseeded initial weights. Fixed (seed before building the model, plus a test), then D3 was re-run from a clean commit. The earlier runs are superseded; the first one was not committed and the second was deleted. *Alternative:* keep the first run's deep numbers. *Why:* only the fixed code is reproducible.
2. **Best deep model = GRU** (macro-F1 0.788 ± 0.014 vs CNN 0.786 ± 0.016), by the rule "highest mean pooled macro-F1". *Alternative:* CNN, which is equally good within noise. *Why:* follows a fixed rule instead of a judgement call; GRU also trains about twice as fast here.
3. **Motion-only features:** each primary channel has its per-sample mean over real rows subtracted, then std, min and max of the centered channels, plus magnitude features of the centered gyro and acc vectors. Roll and pitch *changes and paths* use the uncentered angles, because they are differences and already ignore the mean. *Alternative:* centre the pitch sin/cos and compute pitch from them, which gives meaningless angles. *Why:* every feature stays exactly invariant to a per-sample constant offset (unit-tested), which is the point of the check.
4. **Uncalibrated probabilities are primary** (owner's decision); the temperature-scaled numbers are reported as "did not transfer".

### D6
5. **Row stride** keeps window rows whose position relative to the peak is a multiple of k. The peak row is always kept and the window covers the same span with k times fewer rows. *Alternative:* stride the raw stroke from its first row (the peak could be dropped). *Why:* compares like with like and keeps the alignment of D2.
6. **Window 75% and 50%** keep the peak at the same relative position (36 rows with 12 before the peak; 24 rows with 8 before). *Alternative:* centre the shorter window on the peak. *Why:* the same shape as the 48-row window chosen on coverage in D2.

7. **D6 GRU conditions limited** to full data, every 4th row and motion-only (owner's instruction); the RF covers every condition.
8. **D6 figure labels:** the committed D6 run's figure and report say "Every 2th row" (a label bug, fixed in code afterwards; `results/SUMMARY.md` shows "2nd"). The numbers are unaffected; the run was not repeated just for the label.

### D4
9. **GRU predicts the 6 targets as 6 outputs of one network** (joint MSE on standardized targets) rather than 6 separate networks. *Alternative:* 6 GRUs, six times the compute. *Why:* each criterion still gets its own prediction and metrics; the RF uses fully separate models, so the stronger "separate" reading is covered by the RF.
10. **Per-class baseline uses the true stroke class** of the test stroke. *Alternative:* the predicted class from D3. *Why:* the true class makes the baseline stronger, the conservative direction; the verdict is judged against the level-only baseline anyway.
11. **Baselines use all training players (fit + validation); models use the fit set.** *Why:* the strongest baselines and the D3 model protocol unchanged. Both choices make it harder, not easier, for the models to look good.

### D5
12. **Cross-group forests use the fit set only** (validation players from the training group are set aside but unused by the RF), so within-group and cross-group are trained the same way as in D3/D4. *Alternative:* fit on all training-group players for cross-group. *Why:* the drop from within-group to cross-group stays a like-for-like comparison.
13. **Recording-date table added to `docs/DATASET.md`** (D1 card code extended) after finding that level is confounded with recording day. *Why:* it is a dataset fact that every later level result depends on.

### D8
14. **Confidence is returned uncalibrated** and flagged (`confidence_is_calibrated=False`). *Alternative:* apply a temperature from the inner validation players. *Why:* D3 showed that temperature made calibration worse on unseen players.
15. **The quality estimate stays in the interface** with a note that it mostly reflects player level (D4). *Alternative:* drop it. *Why:* the spec asks for it, and the note makes its limits explicit; easy to remove.
16. **Deployable models are the random forests** (stroke type and quality), retrained on all used strokes. *Alternative:* the GRU. *Why:* the RF had the best D3 macro-F1, is deterministic, and needs no scaler or early stopping.
17. **Latency rerun** after a CLI-only change, so `run.json` points to a clean commit; the first latency run was deleted.

## Open questions for the owner
1. **Calibration across players:** temperature fitted on 2 inner validation players made ECE worse on every model. A temperature fitted on more players (for example, inner leave-one-player-out over all training players) might transfer better, but that changes the inner protocol, so it was not done.
2. **Leak check flagged:** mean orientation alone matches the full models. Separating "real racket-angle difference" from "recording-block artifact" would need data the dataset does not have, such as interleaved stroke types within one recording.
3. **Length check close to its threshold** (+0.085 vs 0.10): worth keeping in mind when reading any model that sees the mask.
4. **D6 says almost no data is needed** (accelerometer only at about 2 rows per second matches full data), but only because the task is solved through static racket angle. Whether to present this as a positive sparsity result or as a sign the task is too easy for this dataset is a framing choice for the write-up.
5. **D4 negative result:** no model beats the level-only baseline. Worth deciding whether D8's quality output should be kept in the interface (it is kept, with a note saying it mostly reflects level) or removed.
6. **Level is confounded with recording day** (all novices on 2019-09-25, all pros on other days). The D5 novice-vs-pro result, and the level-only baseline's strength in D4, may partly reflect session differences. Any skill claim should carry this caveat; separating the two would need novices and pros recorded in the same sessions.
