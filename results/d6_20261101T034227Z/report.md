# D6 sparsity ablation: d6_20261101T034227Z

Commit `8ab3698c5b`; device cpu; arm (macOS-26.6.2-arm64-arm-64bit); 633.7 s.

Same protocol as D3: 15 LOPO folds (P06 not a test player), 1 novice + 1 pro inner validation players per fold, CPU, 3 seeds, pooled out-of-fold macro-F1 (mean ± std over seeds). Δ is the per-seed difference from the same model on full data. Full data = primary channels (acc, gyro, roll, pitch sin/cos), every row, 48-row window. The RF uses handcrafted features of the channels in each condition; the deep model is the D3 GRU.

## Random forest (every condition)

| condition | group | macro-F1 | accuracy | Δ macro-F1 vs full |
|---|---|---|---|---|
| Full data (primary, stride 1, 100%) | reference | 0.798 ± 0.007 | 0.808 | - |
| Every 2th row | rate | 0.808 ± 0.003 | 0.819 | +0.010 ± 0.006 |
| Every 4th row | rate | 0.799 ± 0.004 | 0.813 | +0.002 ± 0.009 |
| Accelerometer only | sensors | 0.798 ± 0.003 | 0.808 | +0.000 ± 0.005 |
| Gyroscope only | sensors | 0.735 ± 0.001 | 0.756 | -0.063 ± 0.008 |
| Accelerometer + gyroscope | sensors | 0.801 ± 0.001 | 0.813 | +0.003 ± 0.005 |
| Primary + heading (yaw, mag) | sensors | 0.802 ± 0.005 | 0.814 | +0.004 ± 0.006 |
| Window 75% | window | 0.802 ± 0.004 | 0.812 | +0.004 ± 0.006 |
| Window 50% | window | 0.808 ± 0.002 | 0.820 | +0.011 ± 0.005 |
| Motion-only (each sample's channel means removed) | motion | 0.667 ± 0.006 | 0.697 | -0.131 ± 0.006 |

Reductions ranked by how much they hurt (RF):

1. Motion-only (each sample's channel means removed): -0.131 macro-F1
2. Gyroscope only: -0.063 macro-F1
3. Accelerometer only: +0.000 macro-F1
4. Every 4th row: +0.002 macro-F1
5. Accelerometer + gyroscope: +0.003 macro-F1
6. Window 75%: +0.004 macro-F1
7. Primary + heading (yaw, mag): +0.004 macro-F1
8. Every 2th row: +0.010 macro-F1
9. Window 50%: +0.011 macro-F1

## GRU (full data, motion-only, every 4th row)

| condition | group | macro-F1 | accuracy | Δ macro-F1 vs full |
|---|---|---|---|---|
| Full data (primary, stride 1, 100%) | reference | 0.788 ± 0.014 | 0.792 | - |
| Every 4th row | rate | 0.797 ± 0.004 | 0.800 | +0.009 ± 0.015 |
| Motion-only (each sample's channel means removed) | motion | 0.629 ± 0.021 | 0.631 | -0.159 ± 0.014 |

Figures: `f1_vs_stride.png`, `ablation_deltas.png`.
