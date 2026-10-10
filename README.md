# Lightning Whisper MLX

Whisper transcription for Apple Silicon, with batching, quantized models, and optional speculative decoding.

![Whisper Decoding Speed](./speed_image.png)

The chart records historical results. Current-source speed, accuracy, and memory
comparisons against Whisper CPP and MLX Whisper remain unqualified until
reproducible benchmarks return data.

## Features

- **Batched Decoding** -> Higher Throughput
- **Distilled Models** -> Faster Decoding (less layers)
- **Quantized Models** -> Faster Memory Movement
- **Lossless speculative decoding** -> Optional assistant-model or MTP acceleration

## Installation

Install lightning whisper mlx using pip:

```bash
pip install lightning-whisper-mlx
```

## Usage

Models

```
["tiny", "small", "distil-small.en", "base", "medium", distil-medium.en", "large", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3"]
```

Quantization

```
[None, "4bit", "8bit"]
```

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

- The default `batch_size` is `12`. Conditioned long-form transcription decodes
  sequentially to preserve previous-text context. Use
  `condition_on_previous_text=False` to batch independent windows. Larger batches
  increase memory use; their throughput benefit requires measurement.
- Dense, 4-bit, and 8-bit target variants have distinct model directories. Existing
  downloaded files remain untouched; quantized variants download into their new
  named directories when explicitly selected.
- `segments` contains dictionaries with `start`/`end` in seconds, text, token IDs,
  and decoder scores. `word_timestamps=True` additionally produces aligned words.
- [MTP training](docs/training/mtp-training.md) documents the local-only pipeline.
  Completing the workflow does not establish trained-head quality or speed.

## Credits

- [Mustafa](https://github.com/mustafaaljadery) - Creator of Lightning Whisper MLX
- [Awni](https://github.com/awni) - Implementation of Whisper MLX (I built on top of this)
- [Vaibhav](https://github.com/Vaibhavs10) - Inspired me to build this (He created a version optimized for Cuda)
