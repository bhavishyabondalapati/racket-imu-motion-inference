# D8 inference latency: d8_20261101T143042Z

Commit `0f75c1f1c6`; device cpu; arm (macOS-26.6.2-arm64-arm-64bit); 9.7 s.

- Hardware: Apple M4 Pro, 14 logical CPUs, CPU only.
- Timed: default infer() (no experimental quality): angle encoding, windowing, 47 handcrafted features, RF classifier (300 trees), single thread, one stroke at a time.
- 1000 timed runs over real strokes (after 20 warm-up runs).
- **Median 6.15 ms, p95 6.37 ms** per window (mean 6.15, max 6.91).
