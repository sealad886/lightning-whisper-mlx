# Short-case performance follow-up

Authorized 2026-10-18. Baseline: `76e3f46`. Continue on the current task branch;
no new worktree, publication or merge. Background telemetry is observational and
must not block benchmark execution, as explicitly requested by the user.

The prior two-pass comparison measured pooled speech speedups of 1.299× base and
1.332× tiny, with pooled short speech/music minima of 1.179× and 1.146×. Full
silence-inclusive ratios were 1.389× and 1.410×. All reference quality gates passed.

## Scope and acceptance

Profile short noisy/music cases with the same FP16, compiled encoder, joint
four-step greedy decoder, explicit suppression, English and no-timestamp settings
as the backend comparison. Attribute fixed preparation/encoder/prefill costs
separately from cached generation. Retain only improvements measured against a
preserved baseline, without changing speech thresholds or degrading transcript,
token, timestamp, score, silence, alignment or fallback behavior. Do not implement
new output formats as part of the parity assessment.

## Checklist

- [x] Profile the short cases and identify measured bottlenecks.
- [x] Implement and validate the smallest quality-preserving optimizations;
  record rejected experiments and regression evidence.
- [x] Rerun two passes of the same 11-fixture base/tiny comparison, two warmups and
  seven repetitions, nonblocking telemetry and reversed model order.
- [x] Audit API/CLI word timing and format parity; document benchmark gaps and
  complete independent review, native verification and prepare the local checkpoint.

Native GPU jobs run serially. Independent source/parity reviews can run in
parallel. The final report separates speech-only ratios, silence shortcuts,
per-fixture spread, first-call costs and quality proof limits. Raw experimental
evidence stays outside Git in `short-case-optimization-2026-10-18`.

## Measured design decisions

Matched baseline stage diagnostics on three short fixtures show encoder costs
around 23 ms base / 9 ms tiny, audio preparation 2.5–3.3 ms, prefill 4.2–4.6 ms
base / 2.1–2.5 ms tiny and two cached chunks around 12 ms base / 8 ms tiny. Stage
synchronization perturbs scheduling; ordinary warmed medians are retained too.

First optimize the CPU frontend's artificial zero tail, preserving complete
frame count, reflect boundaries, global dynamic-range clamp and exact values.
Then test deferring the SOT no-speech host read until sampling is scheduled,
retaining the identical float32 probability expression. Do not truncate encoder
context or change projection shapes, precision or speech thresholds to chase a
short-clip ratio. Extra custom Metal is not justified by these stage totals.

Retained implementation: real samples plus a full 400-sample FFT guard, analytical
zero-mel tail reconstruction and one PCM validation per transcription. New frozen
reference tests and real mel comparisons are exact. The integrated suite passes
465 native tests; 64 complete real transcription outputs and eight word-alignment
cases match baseline exactly. The separate SOT diagnostic deferral lost to the
frontend-only policy in three of four model/pass comparisons and was removed.
Independent source review found no blocker. See the
[evaluation and feature-parity report](../evaluation/short-case-optimization.md).

## Final measured disposition

All four backend runs completed under the requested nonblocking telemetry policy.
Pooled speech speedups are 1.343× base and 1.385× tiny; silence-inclusive totals are
1.440× and 1.472×. Target short noisy/music ratios are 1.28–1.32×, and the lowest
pooled speech fixture is 1.290× base / 1.275× tiny. Individual rounds still vary;
no universal minimum is promised. All 616 measured outputs pass reference quality
assessment, with stable text and tokens across 14 repetitions per backend/model.
Independent adjudication confirms timings, identities and WER/CER arithmetic.

Final native verification: 465 tests pass. Wheel contents match all 31 packaged
first-party sources. Word timings are present in the Python API but absent from
the TXT-only CLI; JSON/SRT/VTT/CSV/LRC exporters remain separate product gaps.
The evaluation prioritizes more diverse short speech, annotated timing accuracy,
long-form behavior, actual cold CLI performance and broader models/languages.
No provider mutation, push, merge or release is included.
