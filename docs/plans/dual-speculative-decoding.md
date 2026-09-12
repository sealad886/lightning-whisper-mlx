# Dual speculative decoding implementation plan

The approved plan is tracked in session context. Repository implementation is
organized by testable slices:

1. Reproducible local environment and sampling regression.
2. Offset-aware block inference and transactional caches.
3. Shared filtered greedy verifier.
4. Assistant and MTP proposal paths.
5. Metadata-only manifest, pseudo-label, dataset, and MLX trainer foundations.
6. Benchmarks and documentation.

Unfinished gates: full-cache training, measured benchmark matrix, independent review,
provider fork/PR/release actions, and model publication licensing evidence.
