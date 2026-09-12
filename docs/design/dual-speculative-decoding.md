# Dual speculative decoding design

Status: implemented foundation; benchmark and batch promotion pending.

## Contract

`none`, `assistant`, and `mtp` are explicit selectors. Accelerated modes require
temperature-zero greedy decoding. Target logits are filtered with the existing
Whisper rules before proposal acceptance. A rejected proposal never remains in the
target cache.

## Execution

`Inference` supports prefill, one-token, block, cache-length, truncation, and reset.
`CachedGreedyVerifier` owns target cache commit/rollback. `AssistantProposer` uses
an independent encoder/cache. `MTPProposer` uses target decoder hidden states and
tied vocabulary projection.

## Current bounds

The executable assistant and MTP paths currently require `batch_size=1`. Rectangular
lockstep batching and automatic mode selection are not enabled until benchmark
results prove their value.
