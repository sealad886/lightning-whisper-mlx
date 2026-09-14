"""Compare greedy decoding with explicit local models and <=30-second audio clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .runner import BenchmarkCase, BenchmarkConfig, BenchmarkSuite


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_results(results) -> dict:
    """Require every repetition to match the baseline's tokens and text."""
    baseline = results[0]
    reference = baseline.outputs[0]
    baseline_stable = all(value == reference for value in baseline.outputs)
    comparisons = []
    for result in results[1:]:
        exact = baseline_stable and all(value == reference for value in result.outputs)
        speedup = baseline.median_seconds / result.median_seconds
        comparisons.append(
            {
                "mode": result.case.mode,
                "lookahead": result.case.lookahead,
                "exact_match": exact,
                "median_speedup": speedup,
                "meets_speed_threshold": speedup >= 1.2,
            }
        )
    return {"baseline_stable": baseline_stable, "comparisons": comparisons}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--assistant", type=Path)
    parser.add_argument("--mtp-checkpoint", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new run path")
    if not (args.assistant or args.mtp_checkpoint):
        parser.error("provide --assistant and/or --mtp-checkpoint")
    # Validate local assets before importing MLX or invoking any model loader.
    for directory in (args.target, args.assistant, args.mtp_checkpoint):
        if directory is not None:
            config = (
                "metadata.json" if directory == args.mtp_checkpoint else "config.json"
            )
            for filename in (config, "weights.npz"):
                if not (directory / filename).is_file():
                    parser.error(f"missing local model file: {directory / filename}")
    for path in args.audio:
        if not path.is_file():
            parser.error(f"missing audio file: {path}")
    cases = [BenchmarkCase(model=str(args.target), label="baseline")]
    for mode, artifact in (("assistant", args.assistant), ("mtp", args.mtp_checkpoint)):
        if artifact is not None:
            cases.append(
                BenchmarkCase(
                    mode=mode, model=str(args.target), lookahead=args.lookahead
                )
            )
    config = BenchmarkConfig(
        cases=tuple(cases), warmups=args.warmups, repetitions=args.repetitions
    )

    import mlx.core as mx
    from lightning_whisper_mlx.audio import (
        N_SAMPLES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from lightning_whisper_mlx.decoding import DecodingOptions, decode
    from lightning_whisper_mlx.load_models import load_model
    from lightning_whisper_mlx.mtp import fingerprint_tokenizer, load_mtp_head
    from lightning_whisper_mlx.tokenizer import get_tokenizer

    target = load_model(str(args.target.resolve()), dtype=mx.float16)
    tokenizer = get_tokenizer(
        target.is_multilingual,
        num_languages=target.num_languages,
        language=args.language,
        task="transcribe",
    )
    assistant = None
    head = None
    artifacts = {"target": target._checkpoint_fingerprint}
    if args.assistant:
        assistant = load_model(str(args.assistant.resolve()), dtype=mx.float16)
        assistant_tokenizer = get_tokenizer(
            assistant.is_multilingual,
            num_languages=assistant.num_languages,
            language=args.language,
            task="transcribe",
        )
        if (
            assistant.dims.n_vocab != target.dims.n_vocab
            or assistant.dims.n_mels != target.dims.n_mels
            or assistant.dims.n_text_ctx < target.dims.n_text_ctx
            or fingerprint_tokenizer(assistant_tokenizer)
            != fingerprint_tokenizer(tokenizer)
        ):
            parser.error("assistant is incompatible with target mel/tokenizer/context")
        artifacts["assistant"] = assistant._checkpoint_fingerprint
    if args.mtp_checkpoint:
        head, _ = load_mtp_head(
            str(args.mtp_checkpoint),
            decoder_hidden_size=target.dims.n_text_state,
            vocabulary_size=target.dims.n_vocab,
            base_model_fingerprint=target._checkpoint_fingerprint,
            tokenizer_fingerprint=fingerprint_tokenizer(tokenizer),
        )
        artifacts["mtp"] = {
            name: file_digest(args.mtp_checkpoint / name)
            for name in ("metadata.json", "weights.npz")
        }

    clips = []
    fixtures = []
    for path in args.audio:
        audio = load_audio(str(path.resolve()))
        if not 0 < audio.size <= N_SAMPLES:
            parser.error(f"audio must contain >0 and <=30 seconds: {path}")
        mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=target.dims.n_mels)
        mx.eval(mel)
        clips.append(mel)
        fixtures.append(
            {"sha256": file_digest(path), "seconds": audio.size / SAMPLE_RATE}
        )
    mx.synchronize()

    def runner(case):
        options = DecodingOptions(
            language=args.language,
            temperature=0.0,
            fp16=True,
            speculative_mode=case.mode,
            lookahead=case.lookahead,
            assistant_model=assistant if case.mode == "assistant" else None,
            mtp_checkpoint=str(args.mtp_checkpoint) if case.mode == "mtp" else None,
            mtp_head=head if case.mode == "mtp" else None,
        )
        outputs = []
        for mel in clips:
            result = decode(target, mel, options)
            outputs.append(
                {
                    "tokens": result.tokens,
                    "text": result.text,
                    "language": result.language,
                }
            )
        mx.synchronize()
        return outputs

    mx.reset_peak_memory()
    suite = BenchmarkSuite(config)
    results = suite.run(
        runner,
        metadata={
            "artifacts": artifacts,
            "fixtures": fixtures,
            "language": args.language,
            "dtype": "float16",
            "scope": "encoder-plus-decoder; prepared mel; sequential single-row clips",
            "tokenizer_fingerprint": fingerprint_tokenizer(tokenizer),
        },
    )
    payload = suite.to_dict(results)
    payload["assessment"] = assess_results(results)
    payload["peak_memory_bytes"] = mx.get_peak_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assessment = payload["assessment"]
    return (
        0
        if (
            assessment["baseline_stable"]
            and all(item["exact_match"] for item in assessment["comparisons"])
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
