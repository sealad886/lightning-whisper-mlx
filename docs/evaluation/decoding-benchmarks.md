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
`outputs` retains an independent snapshot for every timed repetition; `output`
is the final repetition. Runners must finish asynchronous work before returning.
Environment fields cannot be replaced by caller metadata. `git_dirty` exposes
working-tree changes; a commit hash alone does not identify a dirty run.

## CLI

The CLI requires an explicit `module:function` runner. Resolving it executes the
selected module's import code. Runner modules must avoid import-time model loading,
downloads, and fixture selection; the adapter cannot enforce this for arbitrary
user-supplied modules:

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

## Local comparison runbook

Status (2026-09-14): **not run; merge blocked**. Other MLX workloads are active.
Andrew selects the uncontended window and held-out assets. No background benchmark
or automatic merge is scheduled. Do not stop competing jobs to obtain a slot.

`benchmarks.compare_decoding` is the concrete local runner. It requires local
`config.json`/`weights.npz` target and assistant folders, and optionally an MTP
folder with `metadata.json`/`weights.npz`. It never selects or downloads models.
Use a previously validated unquantized base/tiny pair first; quantized target and
assistant artifacts are separate buckets and require loader validation first.
Assistant and target must share mel bins, vocabulary/tokenizer, and sufficient
text context. MTP checkpoint fingerprints must match the exact target/tokenizer.

Prepare held-out clips outside the repository: clean speech, noise, silence,
mixed speech/music, and short/near-30-second utterances. Include each language
for which a claim is intended. The runner rejects empty or >30-second clips;
it must not silently truncate long recordings. Record provenance and confirm
speaker/digest separation from training. Do not publish private audio or transcript
JSON in the PR; attach aggregate results with fixture hashes after inspection.

At the agreed window:

1. Confirm competing MLX jobs have ended, stable thermal state, external power,
   and sufficient free unified memory. Record chip/RAM, macOS, power mode,
   environment package versions, model/fixture hashes, and run start/end times.
   Low CPU usage alone does not prove the GPU is idle. Stop if another job starts,
   thermal throttling occurs, or memory pressure rises; retain the aborted record.
2. Pin the final reviewed source revision. Inspect `git status --short` and keep
   output outside the checkout. Existing user edits/bytecode must not be discarded.
   If the checkout remains dirty, preserve its diff and explicitly identify the
   tested source; do not report that run as clean-commit qualification.
3. Run the synthetic native regression suite outside the sandbox first:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m pytest -q
   ```

4. Run a two-warmup, seven-repetition comparison, replacing paths with selected
   assets. The example is a command template, not evidence of execution:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m benchmarks.compare_decoding \
     --target /path/to/local/target \
     --assistant /path/to/local/assistant \
     --audio /path/to/held-out/clean.wav /path/to/held-out/noisy.wav \
     --language en --lookahead 2 --warmups 2 --repetitions 7 \
     --output /path/to/results/assistant-lookahead2-run1.json
   ```

   Add `--mtp-checkpoint /path/to/checkpoint` only for a correctly trained head.
   Old heads trained against the shifted horizon targets must be retrained; a
   schema-compatible old checkpoint is not evidence of corrected training.
   Use fresh output paths: existing files are rejected rather than overwritten.

5. Repeat independent runs for lookahead 1, 2, and 4. Reverse the lookahead run
   order on the next pass and repeat the baseline to expose thermal drift. The
   runner always places baseline first within a comparison, so compare independent
   passes before attributing a small speed change to the decoder. Treat each
   target/quantization/language/fixture set as a separate bucket. Batch size is one;
   speculative batching is explicitly unsupported and excluded from claims.

## Measurement and sign-off contract

The local runner preloads models and prepares mel features outside the timer.
Each timed invocation includes target encoder and decoder for all supplied clips,
plus assistant encoder/decoder when selected. It synchronizes MLX before returning.
It excludes downloads, model loading, audio decoding, and mel preparation. Model
loading is not a speedup result. Peak memory is process-wide over the suite with
all selected artifacts resident, not incremental memory attributable to each mode.

Every repetition stores full token IDs, text, and language per clip. Timestamp
tokens are part of exact-token comparison. Baseline repetitions must agree with
each other; every accelerated repetition must match that baseline. Exit 1 means
an exactness failure; exit 0 alone does not establish a speedup or merge approval.
The JSON separately records the 1.2x median speed threshold for each comparison.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Synthetic correctness | All native regressions pass on final source | 92 passed locally on 2026-09-14; rerun for final reviewed source if changed |
| Lossless clip decode | Every baseline and candidate repetition matches tokens/text/language | Pending real assets/window |
| Performance | At least 1.2x median speedup in each promoted bucket, reproduced across independent passes | Unmeasured |
| Stability | No OOM/crash; stable baseline durations and uncontended runs | Unmeasured |
| Full transcription | Long-form API/CLI run; identical text and segment timestamps, silence handling and fallback | Pending; clip benchmark does not cover this |
| MTP qualification | Correctly aligned training, held-out loss/acceptance, compatible checkpoint, parity/speed results | Pending trained artifact |
| Review/integration | Clean active review sources, required CI/human approvals, no conflicts, final-head check | Pending PR review |

Real benchmark data is a minimum merge gate, not sufficient by itself. A speed
failure keeps that mode experimental and blocks its promotion; never replace a
failed bucket with an aggregate speedup. Keep automatic acceleration disabled.
Training a full cache or publishing weights requires its own approved assets and
licensing evidence; neither is implied by publishing this source PR.
