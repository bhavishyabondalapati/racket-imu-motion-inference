# D8 inference latency: d8_20261101T040445Z

Commit `33ecc3a8a1`; device cpu; arm (macOS-26.6.2-arm64-arm-64bit); 44.3 s.

- Hardware: Apple M4 Pro, 14 logical CPUs, CPU only.
- Timed: infer(): angle encoding, windowing, 47 handcrafted features, RF classifier (300 trees) and 6 RF regressors (300 trees each), single thread, one stroke at a time.
- 1000 timed runs over real strokes (after 20 warm-up runs).
- **Median 40.25 ms, p95 40.88 ms** per window (mean 40.28, max 44.18).
