# D5 skill level and cross-group generalization: d5_20261101T035811Z

Commit `b89ac1ffb8`; device cpu; arm (macOS-26.6.2-arm64-arm-64bit); 311.9 s.

## Part 1: novice vs pro (15-fold LOPO, D3 folds)

Each test player has a single level, so results are pooled out-of-fold metrics (mean ± std over 3 seeds) plus per-player accuracy, not per-fold F1.

| model | seeds | accuracy | balanced accuracy | macro-F1 |
|---|---|---|---|---|
| majority | 1 | 0.479 | 0.406 | 0.324 |
| rf | 3 | 0.775 ± 0.002 | 0.780 ± 0.002 | 0.772 ± 0.002 |
| gru | 3 | 0.751 ± 0.023 | 0.754 ± 0.030 | 0.747 ± 0.026 |

**Per test player: share of strokes assigned the correct level (mean over seeds)**

| test_player | level | n | gru | majority | rf |
|---|---|---|---|---|---|
| N01 | novice | 99 | 0.498 | 0.0 | 0.697 |
| N02 | novice | 42 | 0.944 | 0.0 | 1.0 |
| N03 | novice | 99 | 0.741 | 0.0 | 0.731 |
| N04 | novice | 100 | 0.92 | 0.0 | 0.887 |
| N05 | novice | 80 | 0.808 | 0.0 | 0.883 |
| N06 | novice | 30 | 0.478 | 0.0 | 0.511 |
| N07 | novice | 50 | 0.953 | 0.0 | 0.993 |
| N08 | novice | 50 | 0.807 | 0.0 | 0.76 |
| P01 | pro | 16 | 0.458 | 1.0 | 0.854 |
| P02 | pro | 134 | 0.689 | 1.0 | 0.726 |
| P03 | pro | 60 | 0.867 | 1.0 | 0.633 |
| P04 | pro | 140 | 0.576 | 1.0 | 0.671 |
| P05 | pro | 115 | 0.974 | 1.0 | 0.971 |
| P07 | pro | 150 | 0.978 | 0.0 | 0.94 |
| P08 | pro | 179 | 0.544 | 1.0 | 0.557 |

**Accuracy by stroke type (mean over seeds and folds)**

| stroke | gru | rf |
|---|---|---|
| BASIC | 0.741 | 0.777 |
| PUSH | 0.758 | 0.701 |
| TOPSPIN | 0.765 | 0.857 |

## Part 2: cross-group generalization (RF only)

Within-group LOPO trains on the other players of the same level; cross-group trains on all players of the other level. Validation players (2) always come from the training group; the forests are fitted on the fit set only, as in D3/D4. Pooled over test samples, mean ± std over 3 seeds. Stroke-type macro-F1 averages the classes present in the test group.

| setting | mode | test samples | stroke-type macro-F1 | score_mean MAE | score_mean Spearman |
|---|---|---|---|---|---|
| train novices -> test novices | within-group LOPO | 550 | 0.690 ± 0.013 | 10.27 ± 0.04 | 0.063 ± 0.008 |
| train novices -> test pros | cross-group | 794 | 0.809 ± 0.014 | 25.09 ± 0.11 | -0.071 ± 0.006 |
| train pros -> test novices | cross-group | 550 | 0.767 ± 0.003 | 22.78 ± 0.05 | 0.141 ± 0.026 |
| train pros -> test pros | within-group LOPO | 794 | 0.839 ± 0.007 | 2.75 ± 0.01 | -0.339 ± 0.004 |

**Change from within-group LOPO to cross-group (negative F1 / positive MAE = worse):**

- **Testing on novices, training on pros instead of other novices:** stroke-type macro-F1 +0.076; score_mean MAE +12.51 points; Spearman +0.078.
- **Testing on pros, training on novices instead of other pros:** stroke-type macro-F1 -0.030; score_mean MAE +22.35 points; Spearman +0.269.
