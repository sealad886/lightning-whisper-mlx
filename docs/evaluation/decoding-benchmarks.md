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
The suite runs one invocation of every case per round, rotating the starting case
each round. Every case occupies each timing position once per complete cycle of
rounds; an incomplete final cycle differs by at most one sample per position.
Results retain configured case order, while `execution_order` records warmup/timed
phase, zero-based round, position, case index, and execution index. Each result's
`sample_order` and `sample_metrics` align with its durations and output snapshots.
The public `run_benchmark` function uses the same machinery for one case. Optional
`before_sample`/`after_sample` hooks measure resources outside the timed interval.
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

Status (2026-10-14): **redesigned parity passes; acceleration unqualified**.
The clean redesign checkpoint
`634d41abb16c4112c8b5966212290f81bfbf3f2b` completed all 54 comparisons with
two warmups and seven timed repetitions per mode. The same nine English fixture
buckets, base target/tiny assistant, float16, lookahead 1/2/4, and two independent
passes were used. Every ordinary baseline matches the original recorded tokens,
text, language, and decision scores exactly.

| Measured gate | Original `e476c7d` | Redesign `634d41a` |
| --- | --- | --- |
| Stable baseline tokens and decision scores | 54/54 | 54/54 |
| Exact candidate tokens/text/language, including timestamps | 50/54 | 54/54 |
| Candidate decision-score parity | 28/54 | 54/54 |
| At least 1.2x median speedup | 0/54 | 0/54 |
| Baseline/candidate median-time ratio | 0.549–0.744 | 0.599–0.833 |

The redesigned candidate takes 1.20–1.67x ordinary decoding time; no bucket is
promoted. The median ratio is 0.694. Separate measurement windows do not establish
an absolute before/after speed improvement. Queued verification preserves target
arithmetic but still executes target steps per token while adding assistant work.
Per-invocation MLX active-allocation peaks were 1,227,462,806–1,231,149,210 bytes
with both selected models resident; these are not process RSS or accelerator-only
memory overhead. All comparisons completed without an OOM/crash.

Seven real-model API pairs also passed text/language, segment token/timestamp and
decision-score parity: clean, conditioned long-form, selected clip ranges, word
alignment (62 words), digital silence, speech/music, and forced temperature-0.2
fallback. The ordinary CLI ran offline with existing model files and returned the
same transcript as the clean API case. These are bounded equivalence checks, not
ground-truth accuracy, other-language, arbitrary-threshold or MTP qualification.

Preflight recorded AC power, thermal state nominal then fair (never observed
serious/critical), and adequate free memory. Per-comparison receipts include
background GPU utilization and memory readings. The run resumed after a guard
mistook its own just-finished GPU activity for contention; the completed result
was retained and a bounded cooldown added. Background desktop activity remains
a performance-qualification limitation. No other workload was interrupted.

### Original failure evidence

The October 10–11 results identify clean source revision
`e476c7dcc37e5b0689d007ebd94dd3a1c693e96e`, MLX 0.32.2, Python 3.12.15,
macOS 27.2 arm64, float16 base target and tiny assistant, and English fixtures.
Each comparison contains two warmups and seven timed repetitions per mode.
The matrix covers nine fixture buckets, lookahead 1/2/4, and two passes.

| Measured gate | Result |
| --- | --- |
| Stable baseline tokens and decision scores | 54/54 comparisons |
| Exact candidate tokens/text/language, including timestamp tokens | 50/54 comparisons |
| Candidate decision-score parity | 28/54 comparisons |
| At least 1.2x median speedup | 0/54 comparisons |
| Observed baseline/candidate median-time ratio | 0.549–0.744; candidate takes 1.34–1.82x baseline time |

Lookahead 1 passes exactness and score parity in all 18 comparisons, but is slower
than baseline. Lookahead 2 and 4 both change text and timestamp tokens for the
near-30-second speech/music fixture in both passes. All 26 score-parity failures
involve average log probability; no-speech probability passes throughout. These
results disqualify the measured assistant buckets from promotion. They do not
qualify MTP, other model pairs, quantization, languages, or full transcription.
Saved files do not establish contemporaneous GPU contention or thermal conditions;
they are retained failure evidence, not proof of uncontended performance.

Raw audio, transcripts, model artifacts, result JSON, and reconciliation reports
remain outside Git; only aggregate evidence is recorded here.
Reassess the unchanged raw outputs when resuming; the prepared matrix manifest
still says "not executed" and records an earlier source revision. Actual result
metadata controls the measured revision and result status. No background benchmark
or automatic merge is scheduled. Andrew selects any new uncontended window;
do not stop competing jobs to obtain a slot.

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
   order on the next pass to expose thermal drift. Modes now interleave within
   each round and rotate timing positions; this reduces ordering bias but cannot
   remove thermal drift or contention. Compare independent passes before
   attributing a small speed change to the decoder. Treat each
   target/quantization/language/fixture set as a separate bucket. Batch size is one;
   speculative batching is explicitly unsupported and excluded from claims.

## Measurement and sign-off contract

The local runner preloads models and prepares mel features outside the timer.
Each timed invocation includes target encoder and decoder for all supplied clips,
plus assistant encoder/decoder when selected. It synchronizes MLX before returning.
It excludes downloads, model loading, audio decoding, and mel preparation. Model
loading is not a speedup result. Each timed invocation resets and reads the MLX
allocation high-water mark outside timing. `sample_metrics` records active memory
before/after, peak active allocation, and allocator cache after that invocation.
The peak includes at least the active allocations present before measurement;
resetting the counter does not unload artifacts. All selected target/assistant/MTP
artifacts remain resident for every mode. These are per-invocation MLX allocation
measurements, not isolated mode residency, incremental accelerator cost, or process
RSS. Top-level `peak_memory_bytes` is the maximum timed-sample active peak; warmup
and model-loading peaks are excluded. Cached buffers are reported separately.

Every repetition stores full token IDs, text, language, average log probability,
and no-speech probability per clip. Timestamp
tokens are part of exact-token comparison. Baseline repetitions must agree with
each other; every accelerated repetition must match that baseline. Token/text/
language equality and decision-score parity are separate gates. Missing or
nonfinite scores fail parity. Finite scores use `math.isclose` with absolute
tolerance `1e-5` and relative tolerance `1e-4`, recorded in the assessment. This is
a small float32 reduction/accumulation budget, allowing numeric noise without
demanding bitwise equality; it is a provisional measurement policy, not proof
of equivalence for real models. Even scores inside tolerance can cross a
transcription retry/silence threshold. Full-transcription behavior, especially
near thresholds, must still be qualified separately. Exit 1 means output or
decision-score parity failed; exit 0 alone does not establish a speedup or merge
approval. The JSON separately records the 1.2x median speed threshold.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Synthetic correctness | All native regressions pass on final source | 173 passed on redesign source; independent affected-path checks also passed |
| Lossless clip decode | Every baseline and candidate repetition matches tokens/text/language and passes decision-score tolerance | Passed 54/54 on `634d41a` |
| Performance | At least 1.2x median speedup in each promoted bucket, reproduced across independent passes | Failed: 0/54; measured ratios 0.599–0.833 |
| Stability | No OOM/crash; stable baseline durations and uncontended runs | 54 complete comparisons, stable baseline outputs, no OOM/crash; background GPU activity limits timing qualification |
| Full transcription | Long-form API/CLI run; identical text and segment timestamps, silence handling and fallback | Seven bounded API pairs and ordinary offline CLI passed; broader languages and arbitrary score-threshold boundaries remain unqualified |
| MTP qualification | Correctly aligned training, held-out loss/acceptance, compatible checkpoint, parity/speed results | Pending trained artifact |
| Review/integration | Clean required review sources, required CI/human approvals, no conflicts, final-head check | PR review source retired clean at `e476c7d` in cumulative round 7/10; separate redesign implementation verification and final raw-evidence reconciliation passed; acceleration unqualified, PR remains open |

Real benchmark data is a minimum merge gate, not sufficient by itself. A speed
failure keeps that mode experimental and blocks its promotion; never replace a
failed bucket with an aggregate speedup. Keep automatic acceleration disabled.
Training a full cache or publishing weights requires its own approved assets and
licensing evidence; neither is implied by publishing this source PR.
