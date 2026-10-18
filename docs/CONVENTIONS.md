# Conventions

## Scope and Purpose

This document records non-obvious repository conventions for decoder, training, and
release changes.

## Core Conventions

### Shared verifier (REQUIRED)

**Scope:** speculative decoding implementations.

**Rule:** Assistant and MTP proposers must route proposals through the same target
verification and cache-commit path.

**Rationale:** A single canonical path keeps timestamp filters, rejection rollback,
EOT handling, and greedy exactness consistent.

**Example:** Extend `lightning_whisper_mlx.speculative.CachedGreedyVerifier`; do not
implement a second acceptance loop in a proposer.

### Source-cache references (REQUIRED)

**Scope:** manifest and labeling tooling.

**Rule:** Training records store source paths and digests and never duplicate source
audio into repository artifacts.

**Rationale:** The active speech cache is large and must remain read-only.

## Rationale and Examples

See `docs/design/dual-speculative-decoding.md` and the tests under `tests/`.

## Known Exceptions

Assistant and MTP decode integration currently require single-row decoding; batch
lockstep regrouping remains future work.

## Change History

- 2026-09-12: Added shared-verifier and source-cache conventions.
