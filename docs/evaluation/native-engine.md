# Native engine measurements and reproduction

Date: 2026-10-18. Implementation baseline: `b670a2e`. See the
[design](../design/native-inference-engine.md),
[plan](../plans/native-inference-engine.md) and
[architecture decision](../adr/native-inference-engine.md).

## Result and qualification

The integrated implementation replaces the ordinary inference hot path with
packed projections, fused attention, bounded indexed KV storage and explicit
compiled decoder state. Resident sessions compile encoder and decoder by default.
The maximum practical gain remains the objective; 1.4× is not a ceiling.

Direct persistent Metal whisper.cpp comparisons recorded the following warm
PCM-to-structured-result medians. Each invocation transcribes all 11 fixtures;
each engine has two warmups and seven timed repetitions. Model order reverses
between independent passes, and engines interleave within each pass.

| Model / pass | MLX seconds | whisper.cpp seconds | cpp / MLX |
| --- | ---: | ---: | ---: |
| base / 1 | 1.0860 | 1.3442 | 1.238× |
| base / 2 | 1.1066 | 1.3317 | 1.203× |
| tiny / 1 | 0.6623 | 0.8239 | 1.244× |
| tiny / 2 | 0.6992 | 0.8296 | 1.186× |

These are observations, not release acceptance or a minimum speed guarantee.
Substantial desktop CPU activity was present; uncontended GPU state was not
established. Across individual paired rounds the aggregate ratios ranged from
0.853× to 1.741×. Per-speech-clip median ratios ranged from 1.074× to 1.353×.
The ratio of summed speech-clip medians, excluding silence, was 1.149×/1.204×
for base and 1.203×/1.183× for tiny across the two passes. This different aggregate
must not be substituted for the table's median whole-suite timing.

All nine speech fixtures pass strict relative reference WER/CER in both models,
both passes and every repetition. Outputs are stable. MLX produces empty text on
3-second silence while C++ emits a word. Both emit a word on 29-second silence,
so **the whole-suite quality gate remains failed**. WER/CER are undefined for a
nonempty hypothesis against an empty reference; the assessor uses an explicit
no-hallucination gate for those cases. Derived reassessments preserve the original
reports rather than rewriting failed evidence.

This experiment disables timestamps. With no-speech probability above 0.6 but
average log probability above −1, the existing confidence override retains the
silence hallucination. No threshold was relaxed and no fixture-specific silence
heuristic was introduced. The normal timestamp-enabled MLX path returned empty
text on both silence fixtures. MLX and C++ also differ in frontend padding:
MLX slices actual-duration mel and pads unused log-mel rows with zero, while C++
pads waveform before mel computation. They consume identical PCM and stored
weights, but do not perform identical floating-point frontend operations.

## Configuration and identities

- Apple M3 Pro, 36 GiB unified memory; Python 3.12.15, MLX 0.32.2.
- AC power; no recorded thermal/performance warning; memory pressure level 1.
- Dense float16 multilingual base/tiny, English, greedy temperature zero,
  fixed language, no fallback, VAD off, no previous-recording context.
- C++ source `5670d5c0bbcb148feabef84400a07cfca9aa3b30`: Release, Metal flash
  attention, Accelerate and OpenMP enabled; Core ML/ANE off. A persistent C API
  worker owns its context. The worker clears request history and uses the same
  explicit numeric suppression list as MLX. Native NST filtering is disabled.
- The same stored MLX checkpoints were converted offline to GGML. Conversion
  receipts verify config, weight and tokenizer/mel asset identities; no source
  model was modified. Conversion reconstructs only deterministic sinusoidal
  positional embeddings when absent and validates every expected tensor.
- Common mono 16 kHz float PCM is decoded/preloaded before timing. Model loading,
  conversion, process startup and input upload are excluded. MLX mel preparation
  and C++ frontend execution are included. Parent wall time includes C++ IPC;
  separate service time excludes it. Build, source, model, PCM and option
  fingerprints accompany raw reports, including a hash of the tested dirty source.

The tested explicit-language inference path is unchanged by the subsequent
short-audio automatic-language reuse correction. Formatting, unused-import and
assessor changes do not establish a new performance result. Initial worker v1
runs used mismatched native NST suppression and are retained as diagnostics only;
the table uses v2 with shared numeric suppression.

## Policy selection, batching and regression checks

A rotating policy comparison on base near-30-second speech measured eager
0.3131 s, compiled decoder 0.2867 s, compiled encoder plus decoder 0.2637 s, and
both compiled with one additional queued step 0.2934 s. Transcripts matched.
Both compiled was 1.187× the eager throughput in this experiment. Extra scheduling
remains opt-in. These policy numbers do not compare against C++ or the original
source, and timing under desktop load remains variable.

The final canonical six-policy CLI also completed with matching text across
every repetition: decoder-only compilation measured 1.264× eager throughput and
both compiled 1.204×; both compiled plus pipelining measured 0.995×. The winner
between decoder-only and both compiled varies across these runs. Decoder
compilation is the consistent benefit; no globally optimal encoder policy is
claimed. Public sessions retain both compiled with independent controls.

Separately synchronized stage diagnostics identify cached decoding as dominant:
about 0.191 s across 88 cached steps versus 0.019 s encoder, 0.005 s mel preparation,
0.004 s prefill and 0.065 s residual host/orchestration/postprocessing in one
instrumented compiled invocation. Synchronization perturbs these timings; they
cannot replace ordinary latency. The warmed compiled policy's MLX active peak was
312,564,548 bytes with approximately 239.5 MB allocator cache, excluding startup
and compilation. This is MLX active allocation, not process RSS or isolated GPU
memory. No OOM or crash occurred in the recorded native comparisons.

Three independent recordings, batched through one base session, took median
0.5259 s versus 0.6582 s sequentially: **1.251× throughput**, two warmups/seven
repetitions, rotating order. Normalized text matched in input order and all three
references had zero word errors in both modes. This is a bounded batching result,
not a multiplier to apply to the C++ comparison.

Differential checks against preserved `b670a2e` source cover 11 real fixtures,
conditioned long-form, clip ranges, language detection, alignment and seeded
temperature fallback. On all 11 fixed-language timestamp-enabled fixtures, text,
tokens and segment boundaries match. Average-log-probability reductions differ
slightly; observed clean/clip maxima were about 0.000324. Optimized ordinary
decoding is not bitwise equivalent and thresholds close to a decision boundary
still require qualification. The concatenated long-form case changes segmentation
and wording, improving reference word errors from 6/116 to 4/116; it is not an
exactness pass. Short auto-language detection retains separate encoding because
its padded input differs from the transcription window. Full matching first
windows reuse features. The real corrected auto-language rerun matches baseline
text and tokens. Word alignment reuses encoder output while retaining its
required full-text decoder pass.

The final base/tiny assistant check passes exact tokens, text, language and score
tolerances on short clean and near-30-second noisy clips at lookahead 2, but runs
at 0.754× ordinary throughput. Assistant/MTP remain experimental; no trained MTP
artifact or new automatic accelerator promotion is implied.

The real CLI completed offline with the existing base artifact, exit 0 and the
expected transcript. Its cold process wall time was 7.72 s on this loaded desktop;
this includes interpreter startup, artifact loading and first-use compilation and
has no matched C++ cold comparison.

## Reproduce

Use the repository's local environment on Apple Silicon, with native Metal access
and `ffmpeg` on PATH. Select a fresh output path for every run. First build
whisper.cpp separately in Release mode with Metal; Core ML must be explicitly
disabled for this comparator. Then build the persistent worker:

```bash
python -m benchmarks.cpp_backend --source /path/to/whisper.cpp \
  --build /path/to/whisper.cpp-build --output /path/to/worker
python -m benchmarks.convert_mlx_to_ggml --help
```

The converter accepts an unquantized local MLX model and writes a new GGML file
and receipt; use its explicit model/output arguments shown by `--help`. It never
downloads or overwrites artifacts. Then compare matched checkpoints:

```bash
python -m benchmarks.compare_backends \
  --mlx-model /path/to/mlx-base --cpp-model /path/to/base.bin \
  --worker /path/to/worker --checkpoint base \
  --audio /path/to/clean.wav /path/to/noisy.wav /path/to/silence.wav \
  --references /path/to/references.json --language en --without-timestamps \
  --compile-encoder --compile-decoder --warmups 2 --repetitions 7 \
  --output /path/to/fresh-comparison.json
python -m benchmarks.profile_inference --model /path/to/mlx-base \
  --audio /path/to/near30.wav --output /path/to/fresh-profile.json
```

References are a JSON array of human reference texts in audio order; empty text
is a valid silence reference. Missing references leave accuracy unqualified.
Timestamp-enabled comparisons remain unqualified for timestamp quality without
reference timestamp review. The current external comparator deliberately rejects
quantized/safetensors-only configurations rather than assuming conversion parity;
product loading supports both formats.

Raw results, references, PCM, transcripts and model artifacts remain outside Git
in the local `native-engine-2026-10-18` evidence directory. Relevant receipts are
`cpp-{base,tiny}-final-pass{1,2}.json`, their `-quality-reassessed.json` derivatives,
`policy-matrix.json`, `profile-packed-compiled.json`, `batch-runtime.json`,
`runtime-{baseline,current}.json`, `runtime-final-check.json` and
`speculative-final.json`; `policy-profile-final.json` exercises the canonical
six-policy comparator plus separate stage diagnostics. The package inspection
receipt verifies a fresh wheel
contains the worker source and core modules without Torch code/dependency.

Final native verification: `PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m
pytest -q` completed **306 passed in 7.50 s**. Changed-file Black checks, focused
fatal/unused-import Flake8, whitespace and local Markdown links passed.
Independent Python/code/input-security review found no outstanding source blocker.
The clean wheel contains 40 members, including the C++ worker, and excludes the
deleted Torch implementation and Torch/tqdm requirements. Coverage percentage was
not measured; the environment has no coverage plugin. Physical timestamp accuracy
and broader quality gates remain separate from test-suite success.

## Remaining acceptance limits

Broader acceptance requires an uncontended repeat, additional Apple Silicon
devices, multilingual/English-only/turbo/large/quantized real artifacts, timestamp
references and silence policy work with representative audio. Cold compilation
latency is separate from these warm gains. No universal 1.2× floor, full-model
family accuracy, Core ML/ANE win or production readiness is established here.

Cached decoding and host overhead remain the measured opportunities. Further
custom Metal kernels require a native GPU trace identifying a specific bottleneck
and a quality-valid end-to-end improvement; Python stage time alone is insufficient.
Speculative depth increases are not justified by the losing real measurements.
Release version remains the existing local 0.0.10. New APIs, the MLX runtime floor
and changed `large` alias require an explicit future release assessment; no tag,
registry publication, PR change or merge was performed by this implementation.
