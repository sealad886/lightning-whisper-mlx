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
4. Active: benchmark runbook, documentation and current-head PR review.

Design decisions: conditioned long-form decoding is sequential; independent
windows may batch with conditioning disabled. Generated artifacts never modify
source caches. Legacy ledgers require an explicit backup/migration with supplied
provenance, and incompatible heads require retraining. No compatibility adapters.

Review sources: independent Codex review and configured Copilot on PR #1.
The integrated native suite passed 169 tests on 2026-10-10. Synthetic workflow
execution and source review do not establish real-audio quality or acceleration.
Merge and automatic merge remain blocked pending real benchmark data.
