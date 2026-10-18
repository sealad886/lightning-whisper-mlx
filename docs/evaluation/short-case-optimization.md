# Short-case optimization and feature coverage

Date: 2026-10-18. Baseline: `76e3f46`. See the
[implementation plan](../plans/short-case-optimization.md) and
[previous decoder work](silence-decoder-followup.md).

## Profile, implementation and experimental disposition

Matched short-case profiles use English greedy transcription, FP16, compiled
encoder, joint four-step generation, explicit suppression and timestamps off,
matching the backend comparison. Encoder work dominates: approximately 23 ms for
base and 9 ms for tiny. Audio preparation adds 2.5–3.3 ms; initial decoder prefill
adds 4.2–4.6 ms base / 2.1–2.5 ms tiny. Stage synchronization changes scheduling,
so separately measured ordinary wall times remain the performance metric.

The previous frontend transforms real PCM plus 30 seconds of artificial zeros.
The optimized frontend transforms real PCM plus a 400-sample guard and fills the
remaining zero-only frames analytically. That guard preserves centered reflection
and every frame intersecting real samples. Full output length, mel dtype,
normalization floor and global maximum clamp are retained. The public frontend
validates input; transcription validates once and calls the shared prepared-PCM
core. No encoder context, precision, speech threshold or decoder policy changes.

An interleaved frontend microbenchmark measures 3.23–4.55× improvement on short
inputs and 1.80–1.93× on a near-30-second control. More importantly, two interleaved
end-to-end policy experiments measure frontend-only gains of 1.046×/1.097× base
and 1.071×/1.139× tiny over the preserved implementation. These bundles contain
three short fixtures plus a near-30-second control; complete results match
exactly across all policies and repetitions. Later stage diagnostics measure
prepared mel at roughly 0.35–0.49 ms on the short inputs. Unchanged encoder
timings also vary with host conditions; sequential pre/post totals must not be
attributed entirely to the frontend change.

Deferring the SOT no-speech diagnostic host read was tested separately. It
preserved exact outputs but added no consistent benefit and lost to frontend-only
execution in three of four model/pass comparisons. The experimental code and its
test were removed; raw results and a reproduction patch remain outside Git.
Further context truncation or projection/precision changes were not retained.

## Verification

All 465 native tests pass. New tests compare the optimized mel array exactly to
the frozen full-padded FFT implementation for float16/float32 inputs, 80/128 mel
bins, short and long recordings, hop remainders, arbitrary padding, endpoint
impulses, DC, zero and very quiet inputs. A resource regression demonstrates that
FFT work stops at real samples plus the centered-window guard. Another regression
proves transcription validates PCM once before invoking the prepared core.

All 64 real transcription cases match the previous checkpoint exactly, including
text, tokens, segment boundaries and scores. Eight real word-alignment cases also
match complete baseline outputs across clean/noisy/music-backed short speech and
absolute 2–7 second clip ranges. This covers 38 aligned words per replay; both
replays match, including probabilities. It is regression evidence, not acoustic
boundary accuracy against human annotations.

Independent source review found no blocker. The isolated wheel has 31 packaged
first-party Python/C++ sources, all byte-identical to the tested checkout. No
dependency or version change was needed. Coverage percentage was not measured.

## Backend comparisons

The same 11-fixture base/tiny matrix is rerun in two passes, reversing model order,
with two warmups and seven measured repetitions per backend per pass. Telemetry
is recorded without blocking execution, as requested. Loading/compilation warmup
is excluded; warmed PCM-to-structured-result timing includes C++ IPC. Both sides
use shared checkpoint weights, PCM, suppression IDs and language/decoding options.

Pooled medians use all 14 repetitions per engine/model. Speedup is median C++
time divided by median MLX time. All 616 measured clip outputs enter the quality
assessment; no samples are discarded.

| Model | Speech MLX / C++ median | Speech speedup | Including silence |
| --- | --- | ---: | ---: |
| base | 0.870434s / 1.169406s | 1.343× | 1.440× |
| tiny | 0.555334s / 0.769019s | 1.385× | 1.472× |

| Short fixture | Previous base | Current base | Previous tiny | Current tiny |
| --- | ---: | ---: | ---: | ---: |
| clean-short-5683-32879-0015.wav | 1.205× | 1.290× | 1.146× | 1.294× |
| noisy-short.wav | 1.203× | 1.318× | 1.170× | 1.302× |
| speech-music-short.wav | 1.179× | 1.304× | 1.168× | 1.280× |

All four runs pass the zero-regression per-clip WER/CER gates. Speech WER is
3.83% base and 9.27% tiny for both engines; MLX has no hallucination on either
silence fixture while C++ hallucinates on both. Text and tokens remain stable
across all 14 repetitions per backend/model.

Base speech pass ratios are 1.352× and 1.293×; tiny ratios are 1.357× and 1.393×.
The targeted pooled short cases all exceed 1.2× in this corpus. Individual timing
rounds still vary below that floor, so this is not a universal minimum. The prior
and current external comparisons occurred in different desktop conditions; the
isolated interleaved frontend-policy experiment above supplies causal evidence
for the retained optimization. Do not attribute all external ratio movement to
the patch or generalize to timestamped, multilingual or larger-model workloads.

Machine: M3 Pro, 36 GiB unified memory, macOS 27.2, MLX 0.32.2, Python 3.12.15.
Receipts bind the baseline Git revision plus a dirty-tree source fingerprint to
the tested implementation. Loading, setup, telemetry and cooldown are excluded.

## Feature parity

The current headline benchmark establishes English text-transcription performance.
It does not establish feature parity for word timing, subtitles, VAD or streaming.

| Capability | MLX Python API | MLX transcription CLI | Local whisper.cpp CLI |
| --- | --- | --- | --- |
| Plain text | Yes | TXT files/stdout | Yes |
| Segment timestamps | `segments` in result | Discarded | Yes, including subtitle formats |
| Word timings | `word_timestamps=True`; attention/DTW alignment | No flag; word fields discarded | Experimental token/word timing and model-specific DTW modes |
| Structured JSON transcript | Result dictionaries; no dedicated writer | No JSON option | JSON/full JSON |
| SRT, VTT, CSV, LRC | No built-in writers | No | Yes |
| Translation, beam, sampling, prompts, clip ranges | API options | Not exposed | CLI controls |
| Independent recording batches | `transcribe_many` | Files processed sequentially | Separate processor/server facilities |
| Learned VAD, diarization, streaming/server | No equivalent first-party workflow | No | Optional models and separate examples |

The API alignment path reuses encoder features but performs a full-text decoder
alignment pass plus CPU median filtering/DTW. First use includes Numba compilation;
the observed first aligned base request took about 0.825 s versus 0.051 s on its
second call, including other first-request setup. Those two observations do not
isolate JIT cost. Translation word timing is explicitly warned as unreliable.
See [the alignment implementation](../../lightning_whisper_mlx/timing.py) and
[transcription CLI](../../lightning_whisper_mlx/cli.py). CLI transcript extraction
keeps only `text`; benchmark JSON is not a user-facing transcript exporter.

For whisper.cpp, token-timing, word-splitting and DTW preset modes have different
grouping semantics from MLX word alignment. `--output-words` creates a karaoke
bash/ffmpeg script, not a JSON word list. These modes need an explicit comparison
contract; the current comparison worker disables token timestamps and DTW.

## Next benchmarks, in priority order

1. **Diverse short speech.** The short clean/noisy/music fixtures here share the
   same three-word utterance, `Yes, said Rachel.` Vary utterance length, token
   count, speakers, accents, SNR, music and speech position within 1–10 seconds.
   This avoids optimizing chunk size around one EOT position or phrase.
2. **Segment and word timing quality plus cost.** Separate text-only, segment
   timestamps and word-aligned configurations on both engines. Annotate reference
   boundaries; report matched-word coverage, missed words, median/p95 boundary
   error, monotonicity, overlaps, clip offsets and long-form drift. Report WER
   separately so timing errors cannot hide recognition errors.
3. **Long-form state and reliability.** Several-minute recordings, words crossing
   30-second boundaries, natural/nonconstant silence, prompts, conditioning on/off
   and clip selections. Measure repeated calls, cache reuse and memory growth.
4. **Real CLI and cold startup.** New processes, model loading, first compilation,
   first alignment JIT, WAV/MP3/video decoding and multi-file directory throughput.
   Resident-PCM timing excludes these user-visible costs. Add enough repetitions
   for meaningful tail latency rather than estimating p95 from seven samples.
5. **Model/language/decoding coverage.** English-only models, large-v3/turbo and
   real 128-mel artifacts, quantization, multilingual/CJK speech, translation,
   beam search and fallback. Qualify accuracy alongside each speed/resource gain.
6. **Batching, memory and energy.** Independent-recording batches, long requests,
   active/peak allocation and RSS, repeated model/session creation and sustained
   thermals/energy where telemetry is available. Report aggregate throughput and
   per-record latency separately.
7. **Export acceptance when writers exist.** JSON scalar normalization, Unicode,
   SRT/VTT rounding and numbering, clip offsets, word preservation, stdout,
   collisions and I/O failure behavior. Measure formatting/I/O separately from
   inference; do not describe missing exporters as benchmark-only gaps.

Raw profiles, frozen baseline location, exact quality outputs, rejected
experiments, wheel inspection and backend JSON receipts remain outside Git in
`short-case-optimization-2026-10-18`.
