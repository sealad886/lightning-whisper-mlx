# Decoding benchmarks

Use `benchmarks/benchmark_decoding.py` with an injected repository-specific runner.
Each result records mode, revision placeholder, Python/MLX/platform versions,
repetition count, durations, median, and exact result tokens.

Promotion remains blocked until repeated comparable runs record exact token and
 timestamp equality, no OOM/crash regression, and at least 1.2x median speedup for
the device/model/batch bucket. No benchmark result is included until a real model
and held-out audio fixture are provided.
