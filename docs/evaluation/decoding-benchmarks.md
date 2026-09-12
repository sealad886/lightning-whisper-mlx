# Decoding benchmarks

The benchmark is composable: `benchmarks.runner.BenchmarkSuite` owns timing and
metadata, while callers inject a runner that receives one `BenchmarkCase` and
returns either any JSON-compatible value or a mapping. Mapping results may include
`tokens` and/or `token_count`; token count is derived from `tokens` when omitted.

```python
from benchmarks.runner import BenchmarkCase, BenchmarkConfig, BenchmarkSuite

config = BenchmarkConfig(
		cases=(BenchmarkCase(mode="none", model="tiny", batch_size=1),),
		warmups=1,
		repetitions=3,
)
results = BenchmarkSuite(config).run(lambda case: {"tokens": [1, 2, 3]})
```

Defaults are intentionally conservative: one warmup, three timed repetitions, one
`none` case, lookahead two, and batch size one. Warmups never enter timing metrics.
Each result records actual git revision, Python/platform/MLX versions, repetition
count, per-run durations, median/min/max, and runner output.

## CLI

The CLI requires an explicit `module:function` runner. This prevents accidental
model downloads or hidden data selection during imports:

```bash
python -m benchmarks.benchmark_decoding \
	--runner my_benchmark:run \
	--output results.json
```

The runner must accept one `BenchmarkCase`. Use `--mode assistant` or `--mode mtp`
only when the supplied runner explicitly loads and validates those artifacts.

Promotion remains blocked until repeated comparable runs record exact token and
timestamp equality, no OOM/crash regression, and at least 1.2x median speedup for
the device/model/batch bucket. No benchmark result is included until a real model
and held-out audio fixture are provided.
