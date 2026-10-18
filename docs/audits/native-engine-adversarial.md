# Native engine adversarial audit

Audit date: 2026-10-18. Baseline: `ee184e8a6a40391b5379b79b218a05a13872e048`.
Branch: `codex/dual-speculative-decoding`; existing checkout, no new worktree.

## Scope and completion criteria

Audit the implemented native MLX Whisper engine: artifact loading, decoding,
transcription, word alignment, model lifecycle, CLI output, tests, memory, and
performance evidence. Fix confirmed defects without changing valid decoder
results or introducing speculative quality shortcuts. Run native MLX jobs serially
using `.venv312` (Python 3.12.15, MLX 0.32.2) outside the sandbox. Background
activity is observational and never blocks benchmarks.

Completion requires reproduction and regression proof for each accepted fix,
independent review, full-suite verification, real checkpoint/quality checks, a
bounded model-retention probe, and fresh base/tiny comparisons. New product
features outside the documented API are not implicit remediation requirements.

## Evidence index

Structured measurements live outside the source tree in the task artifact
directory `adversarial-audit-2026-10-18/` (under this task's Codex visualizations).

- `baseline.json`: source/environment/policy inventory. Baseline suite: 465 passed.
- `mlx-static-candidates.json`: scanner candidates, not measured bottlenecks.
- `model-retention-probe.json` and its probe source: 8 eager and 8 compiled model
  lifecycles, weak references, synchronized allocator measurements.
- `quality-current-joint4.json`, `quality-diff.json`, `word-timing-current.json`
  and `word-timing-diff.json`: real pre-audit replay comparisons.
- `backends-{base,tiny}-pass{1,2}.json`, `run-comparisons.py`,
  `summarize-comparisons.py`, `backend-summary.json`: reproducible performance
  matrix, all samples and quality checks retained.
- `verify-wheel.py`, `wheel-verification.json`: isolated source-snapshot wheel
  build and packaged-source identity checks.
- Source contracts: `load_models.py`, `decoding.py`, `transcribe.py`, `cli.py`.
- Regression proof: checkpoint integrity, prefix context, decoding option
  validation, concurrent model holder, transcription contracts, CLI filesystem
  collisions. Detailed final commands and measurements are recorded below.

Codanna's available index covers the user's home, not this repository; no global
index query/rebuild was used. Source inspection was used instead. No `.beads/`
tracker exists, so this document is the live ledger. MLX strict loading semantics
were checked against [official documentation](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.load_weights.html)
and the installed 0.32.2 API.

## Findings and remediation plan

| ID | Severity | Defect / evidence | Smallest fix | State |
| --- | --- | --- | --- | --- |
| F01 | high | Partial checkpoint updates retain random learned tensors; attention packing invents missing query/value biases. RED: 17 failures. | Strict learned-weight validation after packing/quantization; validate optional alignment metadata separately. | fixed |
| F02 | high | Concurrent `ModelHolder` loads can return checkpoint B for request A and tag B with A's identity. Deterministic thread regression fails. | Serialize lookup/load/publication with an RLock and publish a fully loaded local model. | fixed |
| F03 | medium | Default prefix budget yields `[-0:]`, retaining arbitrarily long prefixes; composed prompt can exceed context. | Bound complete initial sequence, reserve a decode position, trim older prompt before recent prefix. | fixed |
| F04 | medium | Model timestamp tokens can extend a selected 30–40 second clip to 60 seconds; word heuristics can escape clip bounds and poison later speech state. | Bound parsed segments and aligned metadata before seek/speech-state updates; omit wholly outside segments. | fixed |
| F05 | high | Case/Unicode-equivalent output names silently overwrite earlier transcripts on macOS filesystems. RED: 3 failures including inaccessible-output isolation. | Preflight actual filesystem identities, including future names, before any transcription. | fixed |
| F06 | medium | Direct decoding accepts zero/negative/fractional sample limits and non-finite/negative temperatures, sometimes after encoding. | Validate numeric options before inference. | fixed |

Each fix preserves the valid public contract. Roll back a patch if valid
checkpoint loading, transcript/token/score parity, existing seek/fallback behavior,
or measured performance regresses without an explained boundary correction.
Dependencies: artifact validation precedes real model tests; decoder and
transcription fixes precede quality/performance replay. CLI work is independent.

Checklist:

- [x] Pin baseline; map source/contracts and reproduce accepted findings.
- [x] Implement and verify all six findings.
- [x] Independently review integrated changes and resolve in-scope findings.
- [x] Run full tests, quality/word replay, memory proof, and base/tiny benchmarks.
- [x] Close ledger with exact evidence, local commits, and bounded residual risks.

Local fix commits: F01 `57c20e9`; F03/F06 `3695691`; F02/F04 `144a770`;
F05 `855dbaa`. Each includes its regression tests. This document and README link
are committed separately as the audit evidence index. No excluded user changes
were staged or discarded.

## Bounded questions and product gaps

- Model/compiled-function retention: not reproduced in the bounded native probe.
  All 16 model weak references were dead after disposal. Compiled residual active
  memory was 172–212 bytes and returned to 36 bytes after sampler/allocator clear;
  cache memory returned to zero. Process RSS high-water is cumulative and is not
  interpreted as live retained memory. This is a small-model lifecycle probe,
  not an unlimited-duration service soak.
- Scalar reads and shape specialization from static inspection are candidates,
  not proven bottlenecks. Existing benchmark boundaries synchronize lazy MLX work.
- CLI currently documents TXT output. Word timestamps exist in the Python API;
  richer exports, streaming, diarization and VAD are separate product work.
- Word interval bounds and replay stability are testable here; absolute alignment
  accuracy requires word-annotated reference audio. A replay cannot establish it.
- Available real assets cover base/tiny. Other model sizes, real quantized
  artifacts, long multilingual corpora and sustained service loads remain bounded
  coverage gaps, not claims of proven parity.
- Greedy timestamped tiny decoding still repeats conditioned text inside a
  silence-containing recording when fallback/compression checks are disabled.
  Boundary clipping removes only segments beyond actual audio; it is not a
  general hallucination detector. Bounded tradeoff: preserve the requested
  decoding policy and quiet-speech fidelity rather than introduce an energy
  cutoff or force fallback. Exact constant-window rejection deliberately includes
  real neighboring samples used by centered STFT, so a speech boundary can make
  an otherwise silent window eligible for decoding. Existing whole-silence
  fixtures remain separately gated. This is accepted residual model-quality risk
  for the explicit greedy/no-fallback configuration, not corrected by this audit.

## Regression and quality proof

All commands use `.venv312/bin/python`; native commands run outside the sandbox
with Homebrew ffmpeg on PATH. No global dependency installation occurred.

```sh
python -m pytest -q -p no:cacheprovider
```

Result after all six fixes: **532 passed in 10.86 seconds**, up from 465 baseline
tests. Focused failing reproductions preceded fixes: checkpoint set 17 failures;
prefix/options set 24 failures; cache concurrency one failure; timestamp set three
failures; CLI filesystem set three failures. All pass in the integrated suite.
Final rerun after formatting: **532 passed in 20.21 seconds**, with no skips.

Checkpoint regression files cover NPZ/safetensors, dense/quantized tensors,
missing/extra/wrong-shaped learned weights, optional alignment masks and metadata
precedence. The lifecycle fixture now preserves integer metadata while varying
floating parameter precision.

Real `quality-replay.py current joint4` exercised 64 base/tiny outputs across
timestamps on/off, all 11 fixtures, three quiet gains, constant DC and mixed
speech/silence. Compared with the pre-audit snapshot, 61 complete outputs match;
63 transcripts match. Explained differences:

- Base mixed recording: final segment end 62.4 becomes actual end 62.14 seconds.
- Tiny medium recording: final segment end 23.84 becomes actual end 23.31 seconds.
- Tiny mixed recording: final retained end 62.44 becomes 62.14; eight repeated
  segments starting after 62.14 are omitted. Their text/tokens disappear together.

`quality-diff.json` retains full before/after evidence. No valid in-range segment
text, tokens or scores changed. `word-timing-replay.py current` repeats eight
real alignment cases twice (38 words per replay, 76 word records): all complete
outputs equal the pre-audit snapshot, including probabilities and absolute
2–7 second clip offsets. `word-timing-diff.json` records these checks.

Independent code/Python/input-handling review found no blocking issue. A separate
review checked timing bounds, seek/state order and the three real replay changes.
CodeRabbit CLI was unavailable; no provider review is claimed. Native test
execution remains separate evidence from static review approval. Coverage
percentage was not measured because coverage tooling is absent from the declared
environment; no percentage claim is made.

Black checks pass for all 11 changed Python files. Flake8's syntax/undefined-name
checks (`--select=E9,F63,F7,F82`) and `git diff --check` pass. A wheel built through
the existing setuptools 84 backend in a temporary source snapshot contains all
31 expected first-party Python/C++ sources byte-identical to this checkout;
no packages were installed and version remains 0.0.10. The wheel is a local build
artifact, not an installed or published release.
Native CLI smoke test using the real local tiny checkpoint and short clean WAV
exits zero and writes `Yes, said Rachel.` to the audit output directory. Temporary
filesystem-name probes leave no files behind.

## Measured performance after fixes

Same 11-fixture matrix as the short-case follow-up, shared checkpoint/PCM,
English greedy FP16, timestamps off, compiled encoder and joint four-step decoder.
Two warmups and seven measured repetitions per engine/pass. Model order is
base/tiny then tiny/base. Pooled medians retain all 14 samples per engine/model;
speech timing sums the nine speech clips in each round. The all-fixture total
also includes two silence clips. Warm resident PCM-to-result scope includes C++
request/response overhead, excludes model loading and compilation warmup.

| Model | Speech MLX / C++ median | Speech speedup | All-fixture speedup |
| --- | --- | ---: | ---: |
| base | 0.978108s / 1.262949s | 1.291× | 1.389× |
| tiny | 0.567669s / 0.775604s | 1.366× | 1.455× |

| Short fixture | base | tiny |
| --- | ---: | ---: |
| clean-short-5683-32879-0015.wav | 1.236× | 1.296× |
| noisy-short.wav | 1.300× | 1.320× |
| speech-music-short.wav | 1.306× | 1.344× |

One base clean-medium clip measures 1.157×; remaining speech clips exceed 1.2×
in their pooled medians. This is not a universal per-file minimum. Base per-pass
speech ratios were 1.440× and 1.294×; tiny 1.374× and 1.365×. Ratio of pooled
medians need not equal the average of pass ratios. All runs record background
activity (`busy`, no exclusive-hardware qualification); none were blocked or
discarded. No causal pre/post speed gain or regression is attributed to these
correctness patches from separate desktop timing windows.

All four runs pass zero-regression per-clip WER/CER gates across **616 measured
clip outputs**. Exact text/tokens are stable across all 14 repetitions per
engine/model. Speech WER: base **12/313 = 3.83%**, tiny **29/313 = 9.27%**, equal
to C++. Base character errors: 24/1344 for both; tiny 42/1344 MLX versus 44/1344
C++. MLX returns empty text on both silence fixtures in every repetition; C++
hallucinates on both. This quality gate is scoped to text recognition, not
word-boundary accuracy or cross-engine token identity.

Runtime source fingerprint for all four runs:
`06d0b345ed8d6c3bdbd5a9b4ef0feeedb0a1e90b13506d5b5aeb2b45b2299fb2`.
It matches the final tested source. Receipts retain baseline Git revision plus
dirty-tree fingerprint, separating source identity from the later local commits.

## Final verification

All six confirmed defects have implemented fixes, regression proof and independent
review. Memory retention candidate: **not reproducible** in the bounded probe.
Feature/quality/coverage limitations above: **accepted residual risk** within the
explicitly bounded scope. No accepted defect is left blocked. No general speed
floor, large-model qualification, export parity or timestamp acoustic accuracy
claim is made. No push, PR, merge, tag, release or deployment is authorized.
