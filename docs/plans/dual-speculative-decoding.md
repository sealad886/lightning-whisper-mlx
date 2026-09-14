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
