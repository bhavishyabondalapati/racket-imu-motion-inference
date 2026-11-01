# D3 stroke-type classification: d3_20261101T032754Z

Commit `ab928dd4f2`; device cpu; arm; 653.3 s.

Protocol: LOPO over 15 test players (not used as test players: P06). In each fold, 1 novice and 1 pro training player form the inner validation set (early stopping, temperature scaling). Metrics pool all out-of-fold predictions; ± is the std over 3 seeds. Classes: BASIC, PUSH, TOPSPIN.

## Results

| model | seeds | accuracy | macro-F1 | ECE (uncalibrated, primary) | ECE after T (did not transfer) | median T |
|---|---|---|---|---|---|---|
| Majority class | 1 | 0.500 | 0.222 | 0.084 | 0.084 | 1.00 |
| Leak check: RF on per-sample channel means | 3 | 0.813 ± 0.003 | 0.803 ± 0.003 | 0.048 ± 0.003 | 0.082 ± 0.007 | 1.31 |
| Motion-only check: RF with each sample's channel means removed | 3 | 0.697 ± 0.005 | 0.667 ± 0.006 | 0.070 ± 0.003 | 0.121 ± 0.006 | 0.99 |
| Length check: RF on real-row count | 3 | 0.314 ± 0.004 | 0.307 ± 0.003 | 0.155 ± 0.004 | 0.029 ± 0.004 | 20.00 |
| RF, handcrafted features (primary channels) | 3 | 0.808 ± 0.006 | 0.798 ± 0.007 | 0.052 ± 0.009 | 0.089 ± 0.002 | 1.16 |
| Heading check: RF + yaw and magnetometer | 3 | 0.814 ± 0.006 | 0.802 ± 0.005 | 0.056 ± 0.011 | 0.084 ± 0.006 | 0.97 |
| 1D CNN (primary channels) | 3 | 0.794 ± 0.008 | 0.786 ± 0.016 | 0.070 ± 0.011 | 0.108 ± 0.007 | 0.84 |
| GRU (primary channels) | 3 | 0.792 ± 0.017 | 0.788 ± 0.014 | 0.105 ± 0.032 | 0.122 ± 0.029 | 0.84 |

## Shortcut checks

- **Heading check:** adding yaw and the magnetometer changes RF macro-F1 by +0.004 on average (per seed: [-0.0003, 0.0115, 0.0021]); 6 of 15 test players improve. Rule fixed before the run: a gain of at least 0.02 counts as a likely shortcut. Result: not flagged.
- **Leak check (per-sample channel means only):** macro-F1 is +0.581 relative to the majority baseline (0.222). Rule: a margin of at least 0.1 counts as well above. Result: **FLAGGED**.
- **Motion-only check (each sample's channel means removed; no threshold fixed in advance):** macro-F1 0.667, which is -0.131 relative to the full handcrafted RF and +0.445 relative to majority. For comparison, the mean-only RF is +0.005 relative to the full RF.
- **Length check (real-row count only):** macro-F1 is +0.085 relative to the majority baseline. Same rule. Result: not flagged.

## Per test player (macro-F1 over the classes that player has, mean over seeds)

| test_player | n | classes | level | RF, handcrafted features | Heading check: RF + yaw and magnetometer | 1D CNN | GRU |
|---|---|---|---|---|---|---|---|
| N01 | 99 | BASIC, PUSH, TOPSPIN | novice | 1.0 | 0.976 | 1.0 | 1.0 |
| N02 | 42 | BASIC | novice | 0.817 | 0.817 | 0.898 | 0.819 |
| N03 | 99 | BASIC, PUSH, TOPSPIN | novice | 0.613 | 0.613 | 0.662 | 0.662 |
| N04 | 100 | BASIC, PUSH, TOPSPIN | novice | 0.684 | 0.7 | 0.637 | 0.708 |
| N05 | 80 | BASIC, PUSH, TOPSPIN | novice | 0.755 | 0.813 | 0.86 | 0.849 |
| N06 | 30 | BASIC, PUSH | novice | 0.979 | 0.807 | 0.996 | 0.992 |
| N07 | 50 | BASIC | novice | 0.99 | 0.99 | 0.983 | 0.969 |
| N08 | 50 | BASIC | novice | 0.591 | 0.701 | 0.69 | 0.747 |
| P01 | 16 | BASIC | pro | 1.0 | 1.0 | 0.989 | 0.989 |
| P02 | 134 | BASIC, PUSH | pro | 0.821 | 0.826 | 0.788 | 0.788 |
| P03 | 60 | BASIC, PUSH, TOPSPIN | pro | 0.556 | 0.556 | 0.866 | 0.809 |
| P04 | 140 | BASIC, PUSH, TOPSPIN | pro | 0.647 | 0.581 | 0.669 | 0.59 |
| P05 | 115 | PUSH, TOPSPIN | pro | 0.956 | 0.978 | 0.945 | 0.959 |
| P07 | 150 | BASIC, PUSH, TOPSPIN | pro | 0.759 | 0.833 | 0.642 | 0.498 |
| P08 | 179 | BASIC, PUSH, TOPSPIN | pro | 0.996 | 0.984 | 0.893 | 0.965 |

## Per level (pooled within level)

| model | level | n | accuracy | macro-F1 |
|---|---|---|---|---|
| RF, handcrafted features (primary channels) | novice | 550 | 0.781 ± 0.006 | 0.737 ± 0.004 |
| RF, handcrafted features (primary channels) | pro | 794 | 0.827 ± 0.007 | 0.819 ± 0.007 |
| 1D CNN (primary channels) | novice | 550 | 0.810 ± 0.013 | 0.770 ± 0.012 |
| 1D CNN (primary channels) | pro | 794 | 0.783 ± 0.011 | 0.775 ± 0.021 |
| GRU (primary channels) | novice | 550 | 0.822 ± 0.014 | 0.775 ± 0.012 |
| GRU (primary channels) | pro | 794 | 0.771 ± 0.025 | 0.768 ± 0.024 |

## Confusion matrices (pooled, summed over seeds)

**RF, handcrafted features (primary channels)**

|  | pred BASIC | pred PUSH | pred TOPSPIN |
|---|---|---|---|
| true BASIC | 1602 (79%) | 42 (2%) | 372 (18%) |
| true PUSH | 9 (1%) | 1081 (99%) | 2 (0%) |
| true TOPSPIN | 340 (37%) | 9 (1%) | 575 (62%) |

**1D CNN (primary channels)**

|  | pred BASIC | pred PUSH | pred TOPSPIN |
|---|---|---|---|
| true BASIC | 1539 (76%) | 26 (1%) | 451 (22%) |
| true PUSH | 9 (1%) | 1083 (99%) | 0 (0%) |
| true TOPSPIN | 335 (36%) | 9 (1%) | 580 (63%) |

**GRU (primary channels)**

|  | pred BASIC | pred PUSH | pred TOPSPIN |
|---|---|---|---|
| true BASIC | 1501 (74%) | 41 (2%) | 474 (24%) |
| true PUSH | 9 (1%) | 1082 (99%) | 1 (0%) |
| true TOPSPIN | 305 (33%) | 9 (1%) | 610 (66%) |

## Folds

| test player | validation players | fit samples | val samples | test samples |
|---|---|---|---|---|
| N01 | N05, P07 | 1015 | 230 | 99 |
| N02 | N03, P07 | 1053 | 249 | 42 |
| N03 | N05, P04 | 1025 | 220 | 99 |
| N04 | N05, P03 | 1104 | 140 | 100 |
| N05 | N04, P08 | 985 | 279 | 80 |
| N06 | N04, P08 | 1035 | 279 | 30 |
| N07 | N03, P07 | 1045 | 249 | 50 |
| N08 | N05, P07 | 1064 | 230 | 50 |
| P01 | N04, P04 | 1088 | 240 | 16 |
| P02 | N03, P08 | 932 | 278 | 134 |
| P03 | N05, P08 | 1025 | 259 | 60 |
| P04 | N01, P03 | 1045 | 159 | 140 |
| P05 | N04, P04 | 989 | 240 | 115 |
| P07 | N05, P08 | 935 | 259 | 150 |
| P08 | N01, P07 | 916 | 249 | 179 |

Figures: `confusion.png`, `reliability.png`. Data: `oof_predictions.csv.gz`, `metrics.json`, `config.yaml`, `run.json`.
