# Native Whisper inference engine

Date: 2026-10-18. Implementation baseline: `b670a2e`.

## Objective and authority

Deliver the highest practical Apple Silicon Whisper performance while preserving
model identity, transcription quality, timestamp behavior, and bounded memory.
The earlier 1.2–1.4x figure is a qualification reference, not an optimization cap.
This change is authorized for local design, implementation, tests, and benchmarks.
It does not authorize release, deployment, or publication. Existing private audio
and model artifacts remain outside Git; no worktree is required.

## Current state

The existing MLX graph, tokenization, transcription orchestration, quantization,
and benchmark timing infrastructure are retained. The 54 recorded comparisons
measure assistant decoding against ordinary MLX, not whisper.cpp. They demonstrate
bounded parity and a slower assistant; they cannot establish external performance.
Ordinary attention materializes score matrices, KV storage concatenates every
step, completion checks synchronize each token, and retries/alignment repeat
encoding. Model residency is a single global slot; the public catalog is incomplete.

## Requirements and acceptance

| ID | Required outcome | Acceptance evidence |
| --- | --- | --- |
| R1 | Direct reproducible comparison with optimized whisper.cpp | Same model, input PCM, decoding and features; persistent warm engines; independent repetitions, identities, variability, outputs and quality recorded |
| R2 | Efficient ordinary attention | Fused MLX attention when scores are unused; explicit alignment path; causal/cache and real-model quality regression checks |
| R3 | Bounded reusable KV state | Capacity independent of valid offset; append, rollback, beam reordering, context boundary and reset tests |
| R4 | Encode a window once across retries and alignment | Reuse matching model/window/dtype features; unknown-language detection reuse only for the same window; unchanged results |
| R5 | Efficient scheduled generation | Pure compiled numerical regions and intentional evaluation boundaries; no unbounded queued graph; benchmark compiled/eager and scheduling alternatives |
| R6 | Complete model/session ownership | Explicit local/HF model resolution, offline operation, official English-only and turbo catalog, per-engine resident model, alignment metadata and task validation |
| R7 | Complete ordinary decoding | Greedy, sampling and beam search share filters/cache semantics; deterministic beam ranking, finished hypotheses, patience and batch tests |
| R8 | Qualify throughput without changing semantics | Independent-window and independent-request batching; per-record prompts/language/fallback remain isolated; latency and throughput reported separately |
| R9 | Streamlined dependencies and honest documentation | Remove unreferenced Torch implementation/runtime dependency, declared MLX API floor, model support and reproducible local commands |
| R10 | Research remains evidence governed | Shared speculative verifier retained; no automatic promotion of assistant/MTP, quantization or custom kernels without quality and measured benefit |

Quality is not defined as byte-identical output between different backends.
Record normalized text, WER/CER against optional human references, omissions,
hallucinations, language, segment timestamps and silence behavior. Internal
lossless speculative modes retain their stronger exact-token/score contract.
Kernel changes require differential tests plus reference quality evidence; do
not weaken score thresholds to hide fallback changes. Quantization and distilled
models are separate quality/performance configurations.

## Components and contracts

### Benchmarking and profiling

Reuse `BenchmarkSuite` scheduling and serialization. Add an explicit backend
comparison entry point with a persistent whisper.cpp worker linked to the user's
local public C API. The worker owns its context across requests; its protocol
accepts canonical mono 16 kHz float PCM and returns structured results. Build
against the selected checkout rather than duplicating private C structs in ctypes.
Models and converted artifacts are fingerprinted separately with a declared common
source checkpoint. Library/build/source identities and actual options are retained.
Do not include process startup in only one backend's warm measurement.

Report warm PCM-to-result time, cold startup separately, and encoder/prefill/decode
profiles separately from uninstrumented end-to-end measurements. Instrumentation
that synchronizes stages must not masquerade as production latency. Outputs and
private transcripts are local. Accept optional references and calculate quality
metrics; absent references mean accuracy is unqualified. Pin VAD, conditioning,
language, beam/best-of, fallback, precision and timestamp options. Reject or label
unsupported comparisons instead of silently choosing different work.

### Attention and cache

Use MLX's built-in scaled dot-product attention first. Explicit score extraction
is requested by alignment only. Maintain float32 softmax/reductions. A cache owner
stores keys/values, logical offset and bounded capacity; updates use MLX slice
updates. Attention sees only valid positions. Self and cross caches remain distinct.
Rollback changes offset without reconstructing the whole valid history. Beam
reordering applies to every resident cache row. Avoid forcing cache objects into
the model parameter tree or retaining old evaluated graphs.

Stage measurements identified cached decoding as the dominant execution region.
Pack self-attention Q/K/V projections and cross-attention K/V projections into
single linear operations. The loader maps the canonical OpenAI projection names
in memory, including compatible quantized rows, without rewriting source artifacts
or retaining duplicate live weights. The numerical graph remains Whisper.

The compiled one-token step receives only tensor caches and a scalar logical
offset, returns updated buffers explicitly, and shares the ordinary decoder's
attention/MLP implementation. Capacity grows in bounded chunks; padded positions
are masked. This avoids capturing mutable Python cache objects or recompiling for
every token position. Compile token filtering/sampling only through the same
canonical timestamp and EOT rules with an explicit logical history length.

### Generation

The decoder prefill and one-token step remain canonical. Compile pure numerical
regions with explicit state where compilation is beneficial. Dynamic shape changes
must not cause a compilation per token. Asynchronous scheduling queues bounded work
before checking completion; generated rows are masked after EOT. Use measured
policies rather than a growing set of unqualified tuning flags. Beam search uses
the same logits filters and inference state, tracks finished hypotheses per audio,
reorders caches by selected parents and scores final sequences consistently.

### Encoded audio and transcription

Each scheduled window owns encoded features. Fallback accepts encoded features
instead of raw mel on each retry. Alignment still runs its full-text decoder pass
but reuses those features. Language detection uses the same initial-window features
when that exact window is transcribed; selected clip offsets must not reuse the
wrong audio. Detection features are reused only for a full first 30-second window;
short/clipped windows retain their own log-mel padding and encoding. Intermediate
references have request/window lifetime, not an unbounded
global content cache. CPU mel remains a valid choice until measurement selects
otherwise; use explicit CPU stream scope rather than mutating default device.

### Engine and model lifecycle

`LightningWhisperMLX` owns its loaded model and execution policy. Resolve named
models through one catalog, and support explicit local/HF artifacts. Downloads
are a resolver responsibility with explicit offline behavior and cache location.
Existing local config/weights directories can be consumed in place; no destructive
artifact migration is needed. If a conversion is necessary, provide a local
backup/atomic conversion tool rather than silently rewriting user weights.
Warm construction must not repeatedly load or hash unchanged weights. Direct
`transcribe_audio` remains available and accepts an explicit resident model.

Catalog entries identify official checkpoint/task capabilities and alignment
heads. Turbo does not support translation. Unknown custom artifacts remain
usable through validated dimensions/config; an unsupported capability produces
a clear error. Independent engine instances must not evict one another.

### Batching and optional research

Preserve existing conditioned-window scheduling. Batch independent requests at
the model boundary with per-row prompt/filter/fallback state; group only compatible
requests. A simple bounded API is preferred over a background server or queue.
Measure active-row waste and memory before introducing dynamic compaction.
Assistant/MTP continue through the shared verifier. True block verification,
custom kernels, and CPU/GPU overlap are experiments with a stop condition: retain
only quality-valid improvements that survive paired end-to-end measurements.

## Alternatives and decisions

Do not rewrite the model or replace it with a distilled checkpoint to claim engine
speed. Do not treat Python versus C++ as the mechanism. Built-in MLX fusion precedes
custom Metal; explicit cache state precedes broad compilation. Persistent C++
worker avoids ABI-sensitive ctypes parameter layouts and unfair process startup.
Whole-file GPU preprocessing and speculative depth increases require profiling.

## Risks, verification and rollback

Main risks are changed floating-point decisions, incorrect cache offsets after
rejection/beam selection, wrong audio reuse at clip boundaries, and incomparable
benchmark settings. Synthetic contracts, real audio differential checks, native
tests and independent review cover these separately. No existing audio/checkpoint
is overwritten. Git checkpoints permit rollback of each source slice. Runtime
optimizations are promoted from actual measurements; failed experiments are
removed and their evidence retained outside source.

## Performance acceptance

Define speedup as median whisper.cpp time divided by median MLX time for the same
configuration. Seek the maximum reproducible gain; record every promoted bucket
and a stated aggregate, never only the best case. Report uncertainty, cold/warm
scope, memory and quality together. A 1.2x per-bucket reference floor is useful but
not a universal promise across all hardware/models. Compare optimized Metal first;
state Core ML/ANE settings explicitly. Broader Apple Silicon claims need additional
devices. Stop optimization when remaining measured bottlenecks lack a reasonable
quality-preserving improvement, not upon reaching 1.4x.
