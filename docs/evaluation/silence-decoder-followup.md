# Silence and decoder follow-up

Date: 2026-10-18. Baseline: `cb82e1d`. This continues the
[native engine evaluation](native-engine.md) and
[implementation plan](../plans/native-inference-engine.md).

## Silence correction

The recorded 29-second silence fixture contains exactly zero PCM. Whisper assigns
high no-speech probability but also enough confidence to its generated word that
the existing log-probability override retains it. Removing that override globally
could suppress confident quiet speech, so the model policy is preserved.

Transcription now proves whether a requested window contains exactly constant
finite PCM, including DC-only audio. Such a window contains no acoustic variation
and is omitted from decoding. The proof includes the real samples within half an
STFT frame on either side, so speech outside a clip boundary that contributes to
the centered transform cannot be accidentally excluded. Artificial right padding
does not enter the proof. There is no epsilon, energy threshold, VAD dependency or
extra model download. File input is decoded once; the same original PCM supplies
mel preparation and the proof. Absolute seeks, conditioning, language detection,
clip ranges and independent-request ordering remain intact.

Seven public regressions failed before the change; 77 focused native checks passed
afterward. The real matrix contains 64 outputs: base/tiny, timestamps enabled and
disabled, the 11 existing fixtures, quiet speech at three gains, DC and a mixed
speech/silence recording. Against the preserved checkpoint, 61 outputs match
exactly and three change only by removing silence hallucinations. Reference word
errors never increase. All 108 comparable non-silence segment scores match
exactly. At the quietest gain, both versions already miss speech; unchanged output
does not establish perfect quiet-speech accuracy.

The gate applies to original PCM transcription, not the mel-only model decoding
primitive. It does not establish general non-speech detection for room tone,
music or nonconstant noise. Those require representative quality evidence and a
separate speech-detection policy. Shared preparation adds finite/mono validation;
the 30-second finite check measured about 0.558 ms and the constant proof about
0.596 ms on the loaded desktop. Repeated finite validation was not redesigned
because it was small relative to decoder time.

## Profile and implementation

Seven real base invocations contain 623 decoder steps. Host profiling attributes
0.1387 seconds of self time to `Inference.forward_block` across those invocations;
the generation loop's self time also includes GPU completion waits. A bounded
one-step GPU capture and a Metal System Trace were collected. The 10.26-second
instrumented process lifetime contains 6,036 target compute encoder intervals,
with 2.492 seconds summed duration, and external application GPU activity.
Shader timeline collection was disabled. These intervals identify submission work
but do not establish exclusive per-kernel bottlenecks or justify custom Metal.

The cached decoder and canonical greedy sampler now share a compiled numerical
region. It returns updated cache/history/completion/scores, avoiding a full-vocabulary
logit result at the Python boundary. Bounded sequential unrolling performs up to
four autoregressive steps before the next host completion check. Each step still
uses the same float32 cast, suppression/timestamp filters, score computation and
EOT masking. Chunks stop at cache-capacity seams and at sample/context budgets;
growing early would change padded attention width. Extra EOT padding is bounded
and removed by existing finalization. Callable storage is model-owned, bounded to
16 entries and cleared when execution is reconfigured.

Resident compiled sessions select joint four-token decoding. Low-level
`Whisper.configure_execution` retains separate/one-step defaults for controlled
experiments; select `joint_decoding=True, decode_chunk_size=1..4` explicitly.
Nonzero-temperature sampling, beam search, speculative decoding and the optional
pipeline retain their previous execution paths. Encoder policy is unchanged.

New native tests cover float16/float32, timestamps, divergent row completion,
short sample tails, context limits, cache growth, rollback and row reordering.
The complete real 64-output matrix with joint four matches joint one exactly,
including all emitted tokens, segment boundaries and scores.

## Paired decoder measurements

These compare the same redesigned engine and PCM inputs, changing only cached
generation policy. Three speech fixtures cover short clean, near-30-second clean
and near-30-second noisy audio. Each policy has two warmups and seven repetitions,
rotating timing order; model order reverses on the second pass. All recorded
outputs match. These runs are diagnostics under desktop load.

| Model / pass | Joint one / separate | Joint two / separate | Joint four / separate |
| --- | ---: | ---: | ---: |
| base / 1 | 0.973× | 1.119× | 1.145× |
| base / 2 | 1.024× | 1.048× | 1.113× |
| tiny / 1 | 1.000× | 1.161× | 1.325× |
| tiny / 2 | 1.020× | 1.121× | 1.159× |

Four-step execution wins in both models and passes; one-step fusion alone does
not establish a consistent benefit. These ratios must not be multiplied by the
earlier external-backend ratios or advertised as uncontended results. First-call
compilation and short-utterance extra-EOT work remain separate tradeoffs.

The canonical profiling CLI now compares nine policies, including separate and
joint 1/2/4. Synchronized joint stage diagnostics evaluate every raw output tensor,
including scores, before stopping their timer. Instrumentation stays outside the
compiled graph. Stage timing is separate from ordinary wall latency and includes
sampling/filtering in joint cached chunks. Joint instrumentation wraps the outer
inference boundary so a cache miss or eviction can compile a fresh pure graph.
Two regressions cover compilation during instrumentation and restoration after
an exception; a real near-30-second replay records 22 chunks and one cached step.

Final verification passes all **383 native tests**. Changed-file formatting,
fatal/unused lint, whitespace and local Markdown links pass. Independent source
review reports no remaining blocker. An isolated wheel build includes 31
first-party Python/C++ sources, all byte-identical to the tested checkout, and
excludes the removed Torch implementation and Torch/tqdm dependencies. Coverage
percentage was not measured; the environment has no coverage plugin.

## Comparison readiness and reproduction

`--require-uncontended` adds bounded, fail-closed OS observations before model
loading and before/after each complete paired timing round. Sampling/cooldown is
outside measured intervals. GPU-client and paging counters bridge the timed gaps;
the benchmark and its C++ worker are excluded from competing-client detection.
Missing telemetry produces exit 3; observed contention produces exit 4. Both
write a fresh failure receipt. The default diagnostic mode records observations
but cannot qualify a speed claim.

The recorded default policy is three samples spaced 0.5 seconds apart after a
two-second cooldown, GPU utilization at most 5%, total CPU busy at most 20%, known
competing compute processes at most 5% of one core, memory headroom at least 10%,
AC power, nominal thermal state and no active paging. WindowServer alone receives
the explicit GPU background budget. Current Foundation thermal state is observed;
absence of historical `pmset` warnings is not treated as nominal state.

Passing means **observed low contention**, with the sampling policy and limitations
recorded. It cannot prove exclusive hardware ownership, observe every transient
client or exclude all memory-bandwidth interference. No external job is stopped,
suspended or reprioritized by this tool.

The first required-quiet attempt stopped **before model loading**, exit 4:
GPU samples were 17%, 21% and 25%; Foundation thermal state was FAIR (1);
aggregate CPU busy was 99.62% and 98.98%. The snapshots identify active graphify,
multiple backup workers, Finder and external Metal clients. Telemetry was
available: this is an observed-busy result, not missing instrumentation. No new
uncontended external-backend speed number is claimed.

After native tests and packaging finished, a second attempt again stopped before
model loading, exit 4. GPU samples fell to 4%, 0%, 0% and thermal state was nominal,
but CPU busy remained 54.25% and 38.71%. Four swapins and external Codex Service
and NotificationCenter Metal activity were observed. Telemetry was available.
The quiet-window request remains pending; controlled external comparisons remain
blocked. No threshold was relaxed and no external process was altered.

```bash
python -m benchmarks.compare_backends \
  --mlx-model /path/to/mlx-base --cpp-model /path/to/base.bin \
  --worker /path/to/worker --checkpoint base \
  --audio /path/to/clean.wav /path/to/noisy.wav /path/to/silence.wav \
  --references /path/to/references.json --language en --without-timestamps \
  --compile-encoder --compile-decoder --joint-decoding --decode-chunk-size 4 \
  --require-uncontended --warmups 2 --repetitions 7 \
  --output /path/to/fresh-comparison.json
python -m benchmarks.profile_inference --model /path/to/mlx-base \
  --audio /path/to/near30.wav --warmups 2 --repetitions 7 \
  --output /path/to/fresh-profile.json
```

Silence-inclusive direct ratios must explicitly attribute MLX's constant-PCM
rejection: it avoids valid work and improves quality, but does not prove faster
decoder execution on equal evaluated windows. Always retain separate speech-only
ratios and the above controlled-policy comparison.

Raw traces and JSON stay outside Git in the local
`native-followup-2026-10-18` evidence directory. Relevant files include
`host-profile-baseline.json`, `cached-step-baseline.gputrace`,
`baseline-metal-system.trace`, exported interval XML and summary,
`quality-baseline.json`, `quality-current-joint{,4}.json`,
`joint-chunks-{base,tiny}-pass{1,2}.json`, `pcm-guard-cost.json` and
`profile-final-nine-policies.json`, `profile-joint-outer-native.json`,
`strict-base-attempt{1,2}.json`. The final package inspection receipt and wheel
remain in the task's isolated temporary build directory; no artifact was
published.
