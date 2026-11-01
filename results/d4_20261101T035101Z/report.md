# D4 stroke-quality estimation: d4_20261101T035101Z

Commit `613650d6de`; device cpu; arm (macOS-26.6.2-arm64-arm-64bit); 362.4 s.

Same 15 LOPO folds and inner validation players as D3; 3 seeds for the models. Baselines use all training players; the RF regressor uses the fit set; the GRU uses the fit set with the validation players for early stopping. Scores are the coach averages (0-100) of criteria C1-C5 and their mean. No coach-agreement ceiling exists (individual coach scores are not published).

## Does the model judge stroke quality beyond level?

Rule fixed before the run: a model beats the level-only baseline on a target only if its pooled MAE is lower in every seed.

- **RF regressor (handcrafted primary features):** beats the level-only baseline on 0 of 6 targets.
- **GRU:** beats the level-only baseline on 0 of 6 targets.

## All test samples, pooled

**MAE (lower is better)**

| method | C1 | C2 | C3 | C4 | C5 | score_mean |
|---|---|---|---|---|---|---|
| Baseline: global training mean | 10.63 | 13.40 | 18.86 | 8.30 | 12.94 | 12.24 |
| Baseline: per-class training mean | 10.02 | 12.91 | 18.08 | 8.03 | 12.50 | 11.70 |
| Baseline: level-only (training mean of test player's level) | 5.33 | 7.38 | 12.00 | 3.66 | 5.67 | 4.98 |
| RF regressor (handcrafted primary features) | 10.03 ± 0.03 | 13.17 ± 0.05 | 17.61 ± 0.01 | 6.29 ± 0.02 | 11.69 ± 0.05 | 10.85 ± 0.02 |
| GRU | 9.48 ± 0.25 | 12.18 ± 0.18 | 15.89 ± 0.33 | 6.10 ± 0.09 | 10.71 ± 0.21 | 9.90 ± 0.26 |

**Spearman correlation (higher is better)**

| method | C1 | C2 | C3 | C4 | C5 | score_mean |
|---|---|---|---|---|---|---|
| Baseline: global training mean | -0.850 | -0.858 | -0.824 | -0.848 | -0.843 | -0.829 |
| Baseline: per-class training mean | -0.102 | -0.277 | -0.265 | 0.026 | -0.369 | -0.137 |
| Baseline: level-only (training mean of test player's level) | 0.596 | 0.551 | 0.461 | 0.545 | 0.630 | 0.574 |
| RF regressor (handcrafted primary features) | 0.270 ± 0.000 | 0.246 ± 0.009 | 0.316 ± 0.003 | 0.489 ± 0.003 | 0.358 ± 0.008 | 0.281 ± 0.002 |
| GRU | 0.252 ± 0.021 | 0.236 ± 0.033 | 0.324 ± 0.032 | 0.496 ± 0.024 | 0.339 ± 0.056 | 0.297 ± 0.023 |

## Within level (novices only, pros only)

Within a level the level-only baseline is nearly constant, so its Spearman is not meaningful there; a positive within-level Spearman for a model means it ranks strokes of the same level.

**Novices: MAE**

| method | C1 | C2 | C3 | C4 | C5 | score_mean |
|---|---|---|---|---|---|---|
| Baseline: global training mean | 12.75 | 16.17 | 24.54 | 10.25 | 15.31 | 14.57 |
| Baseline: per-class training mean | 12.26 | 15.74 | 22.83 | 9.08 | 14.83 | 14.14 |
| Baseline: level-only (training mean of test player's level) | 10.72 | 13.33 | 21.93 | 2.10 | 12.25 | 10.41 |
| RF regressor (handcrafted primary features) | 11.05 ± 0.08 | 15.06 ± 0.16 | 19.86 ± 0.05 | 5.12 ± 0.03 | 14.08 ± 0.05 | 12.08 ± 0.06 |
| GRU | 12.07 ± 0.26 | 15.27 ± 0.59 | 20.25 ± 0.81 | 6.50 ± 0.22 | 13.69 ± 0.37 | 12.40 ± 0.46 |

**Novices: Spearman**

| method | C1 | C2 | C3 | C4 | C5 | score_mean |
|---|---|---|---|---|---|---|
| Baseline: global training mean | -0.619 | -0.456 | -0.566 | 0.010 | -0.608 | -0.590 |
| Baseline: per-class training mean | -0.181 | -0.341 | -0.144 | 0.364 | -0.424 | -0.239 |
| Baseline: level-only (training mean of test player's level) | -0.632 | -0.588 | -0.610 | -0.335 | -0.684 | -0.723 |
| RF regressor (handcrafted primary features) | 0.073 ± 0.013 | 0.055 ± 0.009 | 0.352 ± 0.003 | 0.041 ± 0.004 | 0.054 ± 0.012 | 0.130 ± 0.008 |
| GRU | 0.043 ± 0.067 | 0.123 ± 0.069 | 0.340 ± 0.061 | -0.038 ± 0.053 | 0.180 ± 0.077 | 0.169 ± 0.082 |

**Pros: MAE**

| method | C1 | C2 | C3 | C4 | C5 | score_mean |
|---|---|---|---|---|---|---|
| Baseline: global training mean | 9.15 | 11.48 | 14.92 | 6.95 | 11.30 | 10.63 |
| Baseline: per-class training mean | 8.47 | 10.96 | 14.79 | 7.30 | 10.89 | 10.00 |
| Baseline: level-only (training mean of test player's level) | 1.59 | 3.27 | 5.12 | 4.73 | 1.11 | 1.22 |
| RF regressor (handcrafted primary features) | 9.32 ± 0.04 | 11.85 ± 0.03 | 16.04 ± 0.03 | 7.11 ± 0.04 | 10.04 ± 0.05 | 10.00 ± 0.06 |
| GRU | 7.68 ± 0.54 | 10.04 ± 0.54 | 12.88 ± 0.66 | 5.82 ± 0.05 | 8.65 ± 0.33 | 8.16 ± 0.63 |

**Pros: Spearman**

| method | C1 | C2 | C3 | C4 | C5 | score_mean |
|---|---|---|---|---|---|---|
| Baseline: global training mean | -0.336 | -0.508 | -0.491 | -0.648 | -0.203 | -0.246 |
| Baseline: per-class training mean | -0.293 | -0.156 | -0.279 | -0.025 | -0.265 | -0.101 |
| Baseline: level-only (training mean of test player's level) | -0.494 | -0.620 | -0.678 | -0.713 | -0.520 | -0.591 |
| RF regressor (handcrafted primary features) | -0.120 ± 0.006 | -0.172 ± 0.003 | -0.092 ± 0.015 | 0.084 ± 0.011 | -0.074 ± 0.009 | -0.153 ± 0.005 |
| GRU | -0.256 ± 0.083 | -0.272 ± 0.068 | -0.110 ± 0.066 | 0.232 ± 0.018 | -0.219 ± 0.126 | -0.233 ± 0.048 |

## Per test player (score_mean; mean over seeds; Spearman n/a if constant)

| test_player | mae class_mean | mae global_mean | mae gru | mae level_mean | mae rf | spearman class_mean | spearman global_mean | spearman gru | spearman level_mean | spearman rf |
|---|---|---|---|---|---|---|---|---|---|---|
| N01 | 21.165 | 21.318 | 18.164 | 8.667 | 18.602 | 0.299 | n/a | 0.096 | n/a | 0.254 |
| N02 | 11.268 | 14.671 | 19.767 | 4.636 | 9.478 | n/a | n/a | 0.399 | n/a | -0.091 |
| N03 | 4.788 | 3.381 | 6.609 | 12.797 | 9.047 | 0.058 | n/a | 0.18 | n/a | 0.346 |
| N04 | 19.911 | 20.041 | 9.895 | 11.068 | 10.131 | 0.827 | n/a | 0.622 | n/a | 0.596 |
| N05 | 8.718 | 7.123 | 4.356 | 13.508 | 5.203 | -0.073 | n/a | 0.721 | n/a | 0.665 |
| N06 | 12.129 | 12.764 | 16.63 | 2.603 | 11.884 | 0.824 | n/a | -0.276 | n/a | -0.131 |
| N07 | 5.686 | 9.257 | 10.301 | 5.287 | 11.554 | n/a | n/a | -0.315 | n/a | 0.167 |
| N08 | 27.986 | 30.694 | 23.684 | 17.52 | 22.929 | n/a | n/a | -0.311 | n/a | 0.083 |
| P01 | 1.92 | 2.267 | 3.33 | 11.444 | 6.354 | n/a | n/a | 0.236 | n/a | -0.824 |
| P02 | 11.203 | 10.886 | 6.133 | 0.688 | 7.491 | -0.248 | n/a | -0.045 | n/a | -0.074 |
| P03 | 8.863 | 9.966 | 5.413 | 0.511 | 13.263 | 0.128 | n/a | 0.174 | n/a | -0.241 |
| P04 | 9.735 | 10.511 | 9.59 | 0.46 | 8.325 | -0.159 | n/a | -0.71 | n/a | -0.353 |
| P05 | 7.621 | 11.378 | 7.521 | 1.224 | 4.686 | -0.951 | n/a | -0.769 | n/a | -0.586 |
| P07 | 8.128 | 9.372 | 3.335 | 1.455 | 3.429 | 0.873 | n/a | 0.519 | n/a | 0.839 |
| P08 | 13.512 | 12.078 | 14.373 | 1.331 | 21.339 | -0.881 | n/a | -0.125 | n/a | 0.357 |

Figure: `quality_scatter.png`.
