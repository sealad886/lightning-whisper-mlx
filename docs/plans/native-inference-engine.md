# Native inference implementation plan

Design: [Native Whisper inference engine](../design/native-inference-engine.md).
Baseline: `b670a2e`; local implementation authorized 2026-10-18.

## Delivery route and boundaries

Use the medium-feature-change lifecycle with context, requirements, design,
planning, implementation, quality, documentation and independent review. Numerical
and cache changes carry medium/high regression risk. Release work is preparation
only; no publication, merge or deployment is included. Local commits are required
for the authorized cleanup/refactoring. No new worktree is authorized or needed.
External coordination is not applicable. Provider review/integration stays outside
this implementation. Existing trained MTP qualification remains artifact-dependent.

## Checklist

- [x] Inspect clean branch, existing contracts, environments and recorded evidence.
- [x] Implement persistent direct backend comparison, quality assessment and stage profiling.
- [x] Implement fused attention and bounded cache state; preserve verifier rollback and add beam search.
- [x] Reuse exact matching encoded windows across detection, retries and alignment; measure scheduling and compilation policies.
- [x] Complete model/session/catalog/offline ownership and dependency cleanup.
- [x] Implement bounded independent-request batching; qualify a real three-recording bucket.
- [x] Run native correctness, real-model quality and paired performance experiments; keep losing scheduling/speculative policies opt-in.
- [x] Complete final verification, independent review and documentation; checkpoint the coherent implementation locally for handoff.

## Work packages and dependencies

| Work | Owner/write scope | Dependencies | Demonstrable result |
| --- | --- | --- | --- |
| W1 / R1 | Benchmark specialist: `benchmarks/`, backend tests | Existing suite, local cpp headers/library and artifacts | Persistent warm comparison, provenance, quality metrics, aggregate ratios |
| W2 / R2,R3,R7 | Primary: `cache.py`, `whisper.py`, `decoding.py`, related tests | Baseline regression; current MLX APIs | Correct fused attention, bounded offset cache, greedy/beam/verifier behavior |
| W3 / R4,R5 | Primary: `transcribe.py`, `timing.py`, `audio.py`, related tests | W2 interfaces | Single encoding per eligible window, explicit CPU stream, measured scheduled/compiled step |
| W4 / R6,R9 | Model specialist: `lightning.py`, `load_models.py`, catalog, packaging and lifecycle tests | Explicit model injection agreed with W3 | Independent resident sessions, local/offline resolution, full catalog, pruned Torch |
| W5 / R8 | Primary after W2–W4: batched API and tests | Shared engine and per-row inference | Compatible independent requests batch with isolated output state |
| W6 / R1–R10 | Primary plus independent reviewer | W1–W5 | Native suite, real quality/performance receipts, final findings and documentation |

W1 and W4 can execute independently after design. W2 → W3 → W5 is the main code
dependency chain. Native GPU workloads run serially even while edits proceed in
independent files. Review/quality failures re-enter only their affected package.
Effort is uncertain until native profiles are available; no delivery date or
performance multiplier is promised.

## Evidence and decisions

Run the existing native suite before edits. Record environment, model/audio hashes,
power/thermal/competing work when observable. Use local base/tiny assets first and
at least two warmups/seven repetitions in final comparisons, with independent
passes when load permits. Record each tested dirty patch alongside the source SHA.
Use short/near-30-second speech, noise, silence/music and long-form/word-alignment
checks; qualify multilingual/model-family coverage only when actual assets exist.
Cold CLI and warmed resident comparison are different evidence. No ground-truth
reference means output parity can pass while accuracy remains unqualified.

RISK-1 numerical drift: characterize optimized target outputs and protect scores,
fallback/silence decisions and timestamps. RISK-2 cache state: test boundary,
reorder, rollback, reset and divergent row completion. RISK-3 benchmark fairness:
matched options, common PCM/checkpoint and persistent contexts. RISK-4 missing
assets or uncontended GPU: finish code and CPU/static tests, retain explicit
unqualified configurations rather than manufacturing results. RISK-5 resource
growth: bound cache/request batches and record active allocation plus process memory
where available. RISK-6 dependency drift: declared MLX minimum and verified local
version; no global installs.

## Implemented disposition and acceptance limits

W1–W5 are implemented. W6 includes two independent base/tiny direct comparison
passes (11 fixtures, two warmups/seven repetitions), a real three-recording
batching comparison, preserved-source transcription checks, seeded fallback,
beam, alignment, automatic-language and offline CLI execution. The design's
once-per-eligible-window rule excludes short automatic-language inputs because
detection and transcription have different padding. A real rerun after fixing
that eligibility matches baseline text and tokens.

Observed warm direct aggregate ratios are 1.186–1.244×, and batching is 1.251×
over sequential resident requests on the measured bucket. Speech quality passes
strict relative reference WER/CER. Both engines hallucinate on 29-second silence
without timestamps, so whole-suite quality acceptance remains failed. Loaded
desktop conditions also prevent an uncontended performance claim. These are
recorded failures/proof limits, not omitted implementation packages. No claim
extends to every model, device, timestamp, silence threshold or quantization.

Independent Python/code/input-security review found no remaining source blocker;
its earlier findings were corrected and retested. Further custom Metal, trained
MTP, broader artifact/device coverage and representative silence-policy research
are separate measured work, not unimplemented default-engine promises. See
[the evidence report](../evaluation/native-engine.md) for reproduction, variability,
quality results and exact remaining acceptance limits.

Final verification: 306 native tests pass; changed-file Black/fatal-and-unused
Flake8, whitespace and local Markdown links pass. The clean isolated wheel
contains the worker/core modules and excludes the deleted Torch code and
Torch/tqdm dependencies. The final independent review of the padding correction,
quality gate, profiler and documentation found no blocker. Coverage percentage
was not measured because no coverage plugin is installed. Commit identity is
recorded in Git and the handoff; publication/merge/release remain outside scope.

## Completion rule

All identified issues have an implementation, verified experimental disposition,
or explicit asset/environment-bound proof gap. Source implementation and measured
performance acceptance are tracked separately. No task is called fully complete
while required work remains. Final handoff includes exact tests, measured gains,
failed experiments, limitations and local/publication status.
