---
title: "ADR-0001: Native Whisper inference engine"
status: "Accepted"
date: "2026-10-18"
authors: "Project owner and implementation team"
tags: ["architecture", "inference", "performance"]
supersedes: ""
superseded_by: ""
---

# Native Whisper inference engine

## Status

Accepted for local implementation. Architectural acceptance does not establish
accuracy or a performance multiplier for an untested model, device, or workload.
The earlier [dual-decoding ADR](dual-decoding-acceleration.md) remains applicable
to explicitly selected research modes.

## Context

The project must preserve OpenAI Whisper model identity, transcription behavior,
and bounded memory while seeking the highest practical Apple Silicon performance.
The 1.2–1.4x reference is not an optimization cap. Existing assistant-versus-MLX
measurements do not establish performance against optimized whisper.cpp.

Profiling identified ordinary cached decoding as a major cost. Separate attention
projections, materialized attention matrices, growing caches, host scheduling,
and repeated audio encoding provide concrete optimization opportunities. Model
and request ownership must also support repeated and independent transcriptions.

## Decision

1. Keep one faithful MLX Whisper graph. Pack self-attention Q/K/V and
   cross-attention K/V into linear projections, mapping existing checkpoint rows
   in memory. Preserve packed quantization values and source artifacts. Use MLX
   fused attention for ordinary inference and explicit cross-attention scores
   when alignment requires them.
2. Store self-attention K/V in request-owned buffers with a logical offset and
   capacity growing in bounded 64-token chunks. Cached compiled steps receive
   arrays and a scalar offset and return updated arrays explicitly. Mask unused
   capacity; retain canonical rollback, replay, and beam-row reordering. Compile
   by stable capacity rather than token position. Reuse cross-attention K/V.
3. Share numerical token selection, score accumulation, EOT handling, and
   timestamp filters between eager and compiled sampling. The compiled greedy
   sampler uses fixed-capacity history plus explicit logical length; its graph
   cache is bounded. Beam search shares the inference/filter contract and owns
   finished hypotheses independently for each recording.
4. Give encoded features window lifetime and reuse them across eligible language
   detection, fallback subsets, and alignment. Reuse requires identical audio
   samples and padding, model, and precision; equal seek or duration alone is
   insufficient. Alignment retains its full-text decoder pass. Use an explicit
   CPU stream for mel preparation instead of changing the default device.
5. Let each public engine own its resident model, artifact identity, precision,
   and execution policy. Resolve named, local, and Hub artifacts centrally with
   explicit offline behavior. Group compatible independent transcription windows
   cooperatively while preserving per-record prompts, language, seek, fallback,
   and input ordering. Keep conditioned windows sequential within each record.
6. Enable encoder/decoder compilation in the public engine's ordinary policy,
   retaining eager controls for differential measurement. Keep one-step
   pipelining disabled by default: it remains a bounded scheduling experiment
   rather than an established improvement.

## Consequences

- **POS-001:** Fewer projection dispatches, intermediate attention allocations,
  repeated encoder executions, and per-token cache reconstructions target
  measured costs while retaining one model implementation.
- **POS-002:** Explicit cache and window ownership make speculative rollback,
  beam selection, request batching, and model residency independently testable.
- **POS-003:** Source checkpoints remain unchanged; local and offline users do
  not require artifact migration or a Torch inference dependency.
- **NEG-001:** Fused/packed/compiled arithmetic can change floating-point decisions.
  Tests and real-model quality evidence remain required after numerical changes.
- **NEG-002:** Compiled attention covers masked capacity, and capacity/batch changes
  can create additional compiled variants. Compilation and padding costs must be
  measured separately from steady inference.
- **NEG-003:** Compatible-window grouping can leave the GPU underfilled when
  prompts diverge. Throughput and single-request latency need separate evidence.

## Alternatives considered

- **ALT-001 — Retain the prior execution graph:** rejected as the optimization
  direction because it leaves demonstrated cache, projection, and encoder-reuse
  costs. Eager execution remains a useful diagnostic policy.
- **ALT-002 — Promote assistant/MTP decoding:** rejected as the ordinary default.
  Recorded assistant verification adds draft work while still performing target
  work token by token; trained MTP quality requires a compatible artifact.
  Explicit research modes remain available without automatic promotion.
- **ALT-003 — Write custom Metal kernels immediately:** deferred until profiles
  demonstrate an important residual cost that built-in MLX operations cannot
  address. Extra native implementation is not justified by a target multiplier.
- **ALT-004 — Enable asynchronous pipelining universally:** rejected as a default
  because bounded queued work can still waste a final step and has not established
  a consistent end-to-end advantage over compiled ordinary generation.
- **ALT-005 — Claim gains through smaller/distilled or differently quantized
  models:** rejected as engine qualification. These are separate configurations
  with their own quality and performance evidence.

## Implementation and acceptance

- **IMP-001:** Follow the [integrated design](../design/native-inference-engine.md)
  and [implementation plan](../plans/native-inference-engine.md). Cache, projection,
  sampler, batching, lifecycle, and real transcription checks cover different
  contracts; static inspection alone does not qualify numerical behavior.
- **IMP-002:** Compare persistent warm engines on the same canonical PCM and
  checkpoint. Verify converted weights and tokenizer/mel assets, use common
  numerical suppression IDs, and record compiled/scheduling options, build/source
  identities, timing variability, outputs, and memory. Keep cold setup and
  synchronized stage diagnostics separate from uninstrumented warm latency.
- **IMP-003:** Define speedup as whisper.cpp median elapsed time divided by MLX
  median elapsed time for a declared configuration. Qualify text against human
  references with relative WER/CER and silence behavior; timestamp workloads need
  their own checks. Passing speech cases cannot qualify a failing silence bucket.
  Long silence without timestamps currently remains a quality limitation; the
  full corpus has no universal speed qualification. After correcting short-window
  padding reuse, the real automatic-language rerun matches baseline text and tokens.
- **IMP-004:** Seek reproducible quality-preserving improvements beyond 1.4x where
  profiles justify them. Report per-workload and aggregate results with their
  scope. Additional models/devices, beam quality, batching throughput, and optional
  research modes require actual evidence rather than extrapolation.

## References

- **REF-001:** [Native inference design](../design/native-inference-engine.md).
- **REF-002:** [Implementation plan](../plans/native-inference-engine.md).
- **REF-003:** [Dual decoding acceleration](dual-decoding-acceleration.md).
- **REF-004:** [MLX fused attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html),
  [compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html), and
  [KV cache guidance](https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html).
- **REF-005:** [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
