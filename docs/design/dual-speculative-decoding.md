# Dual speculative decoding design

Status: implemented foundation; benchmark and batch promotion pending.

## Contract

`none`, `assistant`, and `mtp` are explicit selectors. Accelerated modes require
temperature-zero greedy decoding. Target logits are filtered with the existing
Whisper rules before proposal acceptance. A rejected proposal never remains in the
target cache.

## Execution

`Inference` supports prefill, one-token, block, cache-length, truncation, and reset.
Cached blocks queue single-token target computations without intermediate host
evaluation. This preserves the arithmetic and cache values of ordinary target
decoding: multi-row float16 decoder calls changed real-model tokens and scores.
Initial prompt prefill remains a single causal decoder call. This design batches
evaluation boundaries, not target matrix rows; a throughput gain must be measured.
`CachedGreedyVerifier` owns target cache commit/rollback. `AssistantProposer` uses
an independent encoder/cache and applies the same history-aware logit filters
while drafting. Draft token selection, timestamp rules, and target row scoring
stay on the device until each proposal/verification block is evaluated. Verification
rows use proposal history; rows after the first rejection never enter committed
history. `MTPProposer` uses target decoder hidden states and
tied vocabulary projection.

## Current bounds

The executable assistant and MTP paths currently require `batch_size=1`. Rectangular
lockstep batching and automatic mode selection are not enabled until benchmark
results prove their value.
