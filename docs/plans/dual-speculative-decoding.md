# Dual speculative decoding implementation plan

The approved plan is tracked in session context. Repository implementation is
organized by testable slices:

1. Reproducible local environment and sampling regression.
2. Offset-aware block inference and transactional caches.
3. Shared filtered greedy verifier.
4. Assistant and MTP proposal paths.
5. Metadata-only manifest, pseudo-label, dataset, and MLX trainer foundations.
6. Benchmarks and documentation.

## Finishing milestones (2026-09-14)

- Implemented: baseline decoder return-contract repair; persistent assistant cache
  synchronization; shared context/sample budget enforcement for assistant and MTP.
- Implemented: correct 80/10/10 split accounting and MTP future-target alignment.
- Implemented: transcription CLI with collision detection and per-file failures.
- Implemented: retained per-repetition benchmark evidence and explicit local
  model/held-out-clip comparison runner. See the
  [benchmark runbook](../evaluation/decoding-benchmarks.md).
- Verified locally after round-two fixes: 92 tests passed, including small synthetic
  MLX models, in 4.08 seconds. This is correctness evidence, not a performance measurement.
- Pending: independent review of the final committed changes and PR publication.
- Blocked on an uncontended machine window and selected assets: real-model
  token/timestamp equality, repeated timing, memory/stability, long-form API/CLI
  acceptance, and MTP checkpoint qualification. Merge stays blocked.

Full-cache training, model weight publication, and licensing evidence remain
separate future work. The source foundation does not imply trained or benchmarked
weights. Existing incorrectly aligned heads must be retrained before qualification.

## Defect remediation and training delivery (2026-10-10)

Scope: repair the confirmed transcription, timestamp, model identity, precision,
audio cleanup, labeling, and sampling defects; connect the local MTP workflow.
Source inputs remain read-only. Real training qualification and benchmarking are
deferred until selected assets and an uncontended machine window are available.

Acceptance:

- Every nonempty selected window contributes its own text and timestamp context;
  all clip ranges and requested temperature schedules are honored.
- Canonical timestamp rules retain a legal next token and enforce monotonic time.
- Word timestamps return actual alignment; segments expose seconds and scores.
- Model variants and precisions remain distinct; artifact replacement invalidates
  the cache; audio failure restores the MLX device.
- Local-only MTP execution connects source manifests, immutable teacher labeling,
  frozen target hidden states, head training, held-out metrics, export and reload.
  Invalid supervision and mismatched labeling resumes fail before success.
- Synthetic public-contract and end-to-end workflow tests pass; real accuracy,
  proposal acceptance, speed and memory remain separate pending evidence.

Milestones:

1. Complete: transcription/decoder contracts and regression proofs.
2. Complete: model/cache identity and failure cleanup.
3. Complete: MTP workflow, provenance and synthetic train/export/reload proof.
4. Complete: benchmark runbook, documentation and independent source review.
   Real benchmark execution and integration remain deferred.

Design decisions: conditioned long-form decoding is sequential; independent
windows may batch with conditioning disabled. Generated artifacts never modify
source caches. Legacy ledgers require an explicit backup/migration with supplied
provenance, and incompatible heads require retraining. No compatibility adapters.

Required review source: independent Codex. The holistic review is clean at
`7469767`. On 2026-10-10 Andrew removed Copilot as a required gate after its quota
failures; those attempts remain historical failures, not clean reviews. Codex
remains retired under the review-loop policy.
The integrated native suite passed 169 tests on 2026-10-10. Synthetic workflow
execution and source review do not establish real-audio quality or acceleration.
The final independent Codex review also covered `e476c7d` in cumulative round 7/10.
The source remains retired under the review-loop policy; its static clean result
does not clear native model acceptance gates.

## Consolidation checkpoint (2026-10-14)

Verified integration destination: `sealad886/lightning-whisper-mlx`, `origin/main`
at `5a4d978bcac88af22e612a3e58dfe07dbb3294ba`. Existing PR #1 contains all 23
outstanding commits at `e476c7dcc37e5b0689d007ebd94dd3a1c693e96e`; no additional
committed work exists on `mtp-speculative-speed-up`. Both configured remotes
advertise the same default tip. One primary worktree, no stashes, and no staged,
unstaged, or untracked source changes were present at discovery. Local environments,
build products, caches, and externally stored benchmark assets are preserved.

The native suite passed all 169 tests on the measured source. Assessment of all
54 saved real-model comparisons found exact output parity in 50/54, decision-score
parity in 28/54, and the required speedup in 0/54. The four exactness failures alter
speech/music text and timestamp tokens at lookahead 2/4 across both passes.
See [measured results and limitations](../evaluation/decoding-benchmarks.md).

The initial head remains blocked by failed real-model parity and performance, with full
transcription and trained MTP qualification still outstanding. Preserve PR #1
and its source branch. Andrew expanded consolidation scope on 2026-10-14 to
implement and benchmark decoder redesign. Do not weaken parity tolerance, promote
a failed bucket, or infer runtime acceptance from synthetic tests.

Redesign acceptance and sequence:

1. Preserve ordinary target arithmetic and cache values in queued verification;
   cover float16/float32, rejection, EOT, and context limits with native regressions.
2. Apply shared logit filters to assistant drafts and keep timestamp decisions and
   block scoring on the device, with batch and suppression regression coverage.
3. Run all 54 local fixture comparisons against a clean source checkpoint, keeping
   old results and raw artifacts outside Git. Record exactness, decision scores,
   timing, memory, source identity, and environmental limitations independently.
4. Qualify full transcription separately. Retain explicit experimental modes and
   block promotion whenever the repeated 1.2x speed requirement is unmet.
5. Reconcile review coverage and integration gates before any merge or cleanup.

Initial diagnostic: queued float16 verification restored exact tokens/text on the
previously failing near-30-second speech/music fixture, with average-logprob drift
about 2.2e-8. Draft filtering reduced verification rounds from 26 to 22. Single
profiles still ran slower than ordinary target decoding; these observations are
diagnostics, not full-matrix qualification or a speedup claim. Isolated float32
and fused-attention experiments did not establish a speedup and were not adopted.

Completed evidence at clean redesign checkpoint `634d41a`:

- All 173 native regressions passed. Independent affected-path verification passed
  35 tests, 54 timestamp-rule comparisons and 108 verifier scenarios.
- All 54 real comparisons passed exact output and decision-score parity. Every
  baseline output and score remains identical to the original recorded baseline.
- Seven real full-transcription API pairs and an ordinary offline CLI run passed;
  long-form conditioning, clip ranges, word alignment, silence and fallback covered.
- Independent reconciliation verified raw results, repetitions/order, model and
  fixture hashes, timing and memory summaries. Original assets/results preserved.
- No comparison reaches 1.2x speedup. Ratios 0.599–0.833, median 0.694. Strict
  queued arithmetic still executes target work per token and adds assistant work.
  Acceleration remains experimental and unpromoted; PR #1 remains open and no
  branch/worktree cleanup or merge is justified by these performance results.

Remaining acceptance: a measured faster exact target execution design, broader
language/model/quantization coverage as claims expand, score-threshold boundaries,
and a separately trained compatible MTP artifact. Do not repeat completed source
inventory, merges or benchmark runs blindly; resume from retained source/result
identities and the measured performance gap.
