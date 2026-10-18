# Lightning Whisper MLX

Whisper transcription for Apple Silicon, with batching, quantized models, and optional speculative decoding.

![Whisper Decoding Speed](./speed_image.png)

The chart records historical results. The native engine now has direct, repeated
Metal whisper.cpp comparisons: observed warm aggregate speedups were 1.19–1.24×
on an M3 Pro with base/tiny at checkpoint `cb82e1d`. These are bounded observations
under desktop load, not a universal minimum. The
[follow-up](docs/evaluation/silence-decoder-followup.md) fixes exact-silence
hallucination, adds faster bounded greedy chunks, and requires recorded contention
checks for new performance qualification. See the
[original measurements](docs/evaluation/native-engine.md) for their precise scope.

## Features

- Fused MLX attention, packed projections and compiled four-token greedy chunks.
- Resident model sessions, offline local artifacts and bounded recording batches.
- Greedy, sampling, beam search and word alignment through the same model graph.
- Optional distilled/quantized artifacts and experimental verified speculation.

## Installation

Install lightning whisper mlx using pip:

```bash
pip install lightning-whisper-mlx
```

## Usage

Named models

```
tiny, tiny.en, base, base.en, small, small.en, medium, medium.en,
large-v1, large-v2, large-v3, large, large-v3-turbo, turbo,
distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3
```

Quantization

```
[None, "4bit", "8bit"]
```

Quantized catalog variants are available for the multilingual and distilled
families; English-only and turbo entries currently select dense artifacts.
`large` selects large-v3; use `large-v1` explicitly for the original model.
English-only, distilled and turbo entries reject translation. Catalog support is
broader than measured runtime coverage: this change was qualified with local
dense base/tiny artifacts.

```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(
    model="/path/to/model",  # config.json + weights.npz or weights.safetensors
    local_files_only=True,
    batch_size=3,
)
result = whisper.transcribe("recording.wav", language="en")
results = whisper.transcribe_many(["one.wav", "two.wav"], language="en")
beam_result = whisper.transcribe("recording.wav", language="en", beam_size=3,
                                  temperature=0.0)
```

Each engine lazily loads and retains its own model. Named/Hub models use the
Hugging Face cache, optionally selected with `cache_dir`; existing local artifacts
are read in place. Encoder and cached decoder compilation are enabled by default.
The first call includes loading/compilation costs. Set `compile_encoder=False,
compile_decoder=False` when constructing an engine to select eager execution.
Compiled greedy sessions combine cached decoding with canonical sampling in
bounded chunks, preserving context limits and EOT scores. Other decoding modes
retain their own execution paths.
The optional `pipeline=True` transcription setting queues one additional step;
it remains disabled by default because it lost the interleaved policy comparison.

Speculative decoding is opt-in and supports temperature-zero greedy decoding.
`none` remains the default. Assistant mode currently requires `batch_size=1` and
an assistant model with a compatible tokenizer vocabulary; MTP mode requires a
versioned checkpoint produced by the local MTP training workflow.

```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(
	model="base",
	speculative_mode="assistant",
	assistant_model="mlx-community/whisper-tiny-mlx",
	lookahead=2,
	batch_size=1,
)
```

Both modes verify every proposed token with the target model. Rejected proposals
are corrected by the target, and existing timestamp/suppression filters remain
canonical. No accelerator is enabled automatically without benchmark evidence.
Speculative modes remain experimental: the local base/tiny English matrix passes
output parity after the decoder redesign, but runs slower than ordinary decoding.
See the [measured results](docs/evaluation/decoding-benchmarks.md) before enabling one.

## Development

Use a project-local Python 3.12 environment on Apple Silicon. Run the test suite
outside sandboxed execution because MLX requires Metal access:

```bash
python -m pytest -q
```

Training manifests reference source cache files and never copy source audio.
Pseudo-label output and MTP checkpoints are generated artifacts and should remain
outside Git unless a small provenance record is explicitly requested.

## Command-line transcription

The `lightning-whisper-mlx` command accepts one file, multiple files, or directories:

```bash
lightning-whisper-mlx recording.mp3
lightning-whisper-mlx one.wav two.m4a
lightning-whisper-mlx recordings/ --recursive --output-dir transcripts/
```

Without `--output-dir`, each transcript is written beside its source as a `.txt`
file. Use `--output-dir ''` (or `--output-dir '""'`) to print transcripts to
stdout. A non-empty output directory preserves relative parent directories, so
same-named files do not overwrite one another.
Inputs in the same directory that share a stem (for example `clip.wav` and
`clip.mp4`) would share a transcript path. The CLI reports those inputs as failures
before transcription; rename them or use stdout. Other files still continue.

Supported video files have their audio stream extracted through `ffmpeg` before
transcription. Install `ffmpeg` separately and ensure it is on `PATH`.

See the [benchmark runbook](docs/evaluation/decoding-benchmarks.md) for the local
comparison command, deferred execution window, and remaining merge gates.

#### Example

```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-medium.en", batch_size=12, quant=None)

text = whisper.transcribe(audio_path="/audio.mp3")['text']

print(text)
```

## Notes

- The default `batch_size` is `12`. `transcribe_many` returns results in input
  order and batches compatible active windows from independent recordings while
  retaining each recording's own prompt, language and fallback state. One engine
  serializes simultaneous callers; use the batching API for shared-model work.
  Conditioned long-form transcription decodes
  sequentially to preserve previous-text context. Use
  `condition_on_previous_text=False` to batch independent windows. Larger batches
  increase memory use; their throughput benefit requires measurement.
- Dense, 4-bit, and 8-bit target variants have distinct model directories. Existing
  downloaded files remain untouched; quantized variants download into their new
  named directories when explicitly selected.
- `segments` contains dictionaries with `start`/`end` in seconds, text, token IDs,
  and decoder scores. `word_timestamps=True` additionally produces aligned words.
- Transcription skips windows proven to contain exactly constant finite PCM,
  including digital silence and DC-only audio, without an amplitude cutoff.
  Quiet nonconstant speech is retained. This is not general VAD for background
  noise or music; the mel-only decoding primitive keeps the model's raw behavior.
- [MTP training](docs/training/mtp-training.md) documents the local-only pipeline.
  Completing the workflow does not establish trained-head quality or speed.
- The [engine design](docs/design/native-inference-engine.md),
  [implementation plan](docs/plans/native-inference-engine.md) and
  [benchmark runbook](docs/evaluation/native-engine.md) describe the graph,
  decisions, measured scope and remaining qualification work.

## Credits

- [Mustafa](https://github.com/mustafaaljadery) - Creator of Lightning Whisper MLX
- [Awni](https://github.com/awni) - Implementation of Whisper MLX (I built on top of this)
- [Vaibhav](https://github.com/Vaibhavs10) - Inspired me to build this (He created a version optimized for Cuda)
