# ADR: dual decoding acceleration

- Status: Accepted
- Date: 2026-09-12

## Context

Whisper decoding is autoregressive, while Apple Silicon benefits from reducing
sequential target calls. Proposals must not change deterministic greedy output.

## Decision

Support explicit assistant-model and in-model MTP proposers behind one target-owned
filtered verifier with transactional cache rollback. Keep `none` as default and do
not add automatic selection without repeated benchmark evidence.

## Consequences

The verifier is the correctness boundary. Assistant mode uses more memory; MTP mode
requires separately trained compatible heads. Current accelerated integration is
single-row until rectangular lockstep behavior is measured and implemented.
