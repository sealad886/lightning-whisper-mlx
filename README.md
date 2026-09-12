# Lightning Whisper MLX

An incredibly fast implementation of Whisper optimized for Apple Silicon.

![Whisper Decoding Speed](./speed_image.png)

10x faster than Whisper CPP, 4x faster than current MLX Whisper implementation.

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
versioned checkpoint produced by the training tools.

```python
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

#### Example

```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-medium.en", batch_size=12, quant=None)

text = whisper.transcribe(audio_path="/audio.mp3")['text']

print(text)
```

## Notes

- The default batch_size is `12`, higher is better for throughput but you might run into memory issues. The heuristic is it really depends on the size of the model. If you are running the smaller models, then higher batch size, larger models, lower batch size. Also keep in mind your unified memory!

## Credits

- [Mustafa](https://github.com/mustafaaljadery) - Creator of Lightning Whisper MLX
- [Awni](https://github.com/awni) - Implementation of Whisper MLX (I built on top of this)
- [Vaibhav](https://github.com/Vaibhavs10) - Inspired me to build this (He created a version optimized for Cuda)
