# Review queue

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

## Open questions for the owner
1. **Calibration across players:** temperature fitted on 2 inner validation players made ECE worse on every model. A temperature fitted on more players (for example, inner leave-one-player-out over all training players) might transfer better, but that changes the inner protocol, so it was not done.
2. **Leak check flagged:** mean orientation alone matches the full models. Separating "real racket-angle difference" from "recording-block artifact" would need data the dataset does not have, such as interleaved stroke types within one recording.
3. **Length check close to its threshold** (+0.085 vs 0.10): worth keeping in mind when reading any model that sees the mask.
