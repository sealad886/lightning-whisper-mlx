"""Offline, persistent warm PCM-to-result comparison of MLX and Metal whisper.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time

from .cpp_backend import CppBackend, MAX_SAMPLES, file_digest, source_identity
from .quality import assess_backend_results
from .runner import BenchmarkCase, BenchmarkConfig, BenchmarkSuite
from .contention import ContentionError, ContentionPolicy, QuietGuard


def decode_pcm(path):
    """Canonical ffmpeg mono 16 kHz float PCM; preparation is outside warm timing."""
    import numpy as np

    with tempfile.TemporaryFile() as output:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-t",
                str(MAX_SAMPLES / 16000 + 1 / 16000),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-",
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            check=True,
            timeout=120,
        )
        if not 0 < output.tell() <= MAX_SAMPLES * 4:
            raise ValueError("audio must be nonempty and <=30 minutes")
        output.seek(0)
        raw = output.read()
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ValueError("audio contains nonfinite samples")
    return audio


def canonical_suppression_ids(tokenizer, n_vocab):
    """Default MLX -1 suppression, including decoding's always-suppressed specials."""
    ids = set(tokenizer.non_speech_tokens)
    ids.update(
        (
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        )
    )
    if tokenizer.no_speech is not None:
        ids.add(tokenizer.no_speech)
    if any(not isinstance(token, int) or not 0 <= token < n_vocab for token in ids):
        raise ValueError("tokenizer suppression IDs are outside model vocabulary")
    return tuple(sorted(ids))


def model_suppression_ids(model, language, task):
    from lightning_whisper_mlx.tokenizer import get_tokenizer
    from lightning_whisper_mlx.decoding import DecodingTask
    from types import SimpleNamespace

    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )
    ids = canonical_suppression_ids(tokenizer, model.dims.n_vocab)
    # Check against the canonical implementation, avoiding a second maintained policy.
    bound = SimpleNamespace(
        tokenizer=tokenizer, options=SimpleNamespace(suppress_tokens="-1")
    )
    if ids != DecodingTask._get_suppress_tokens(bound):
        raise ValueError("benchmark suppression policy differs from canonical decoder")
    common = tuple(sorted(set(ids).union(tokenizer.all_language_tokens)))
    if any(not 0 <= token < model.dims.n_vocab for token in common):
        raise ValueError("language suppression IDs are outside model vocabulary")
    return common


def verify_conversion_assets(conversion, config, *, assets=None):
    assets = Path(
        assets or Path(__file__).resolve().parents[1] / "lightning_whisper_mlx/assets"
    )
    tokenizer = (
        "multilingual.tiktoken" if config["n_vocab"] >= 51865 else "gpt2.tiktoken"
    )
    runtime = {
        "tokenizer": file_digest(assets / tokenizer),
        "mel_filters": file_digest(assets / "mel_filters.npz"),
    }
    if conversion.get("assets") != runtime:
        raise ValueError(
            "GGML conversion mel/tokenizer assets differ from MLX runtime assets"
        )
    return runtime


def validate_artifacts(args):
    config = json.loads((args.mlx_model / "config.json").read_text())
    if config.get("quantization") is not None:
        raise ValueError("matched baseline currently requires unquantized MLX")
    fingerprints = {
        name: file_digest(args.mlx_model / name)
        for name in ("config.json", "weights.npz")
    }
    with args.cpp_model.open("rb") as stream:
        header = stream.read(48)
    if len(header) != 48:
        raise ValueError("truncated GGML model")
    magic, *values = struct.unpack("<12i", header)
    if magic != 0x67676D6C or values[-1] != 1:
        raise ValueError("comparison requires unquantized f16 GGML model")
    dimensions = (
        "n_vocab",
        "n_audio_ctx",
        "n_audio_state",
        "n_audio_head",
        "n_audio_layer",
        "n_text_ctx",
        "n_text_state",
        "n_text_head",
        "n_text_layer",
        "n_mels",
    )
    if any(config[key] != value for key, value in zip(dimensions, values[:-1])):
        raise ValueError("backend model dimensions differ")
    if config["n_vocab"] < 51865 and (
        args.language != "en" or args.task != "transcribe"
    ):
        raise ValueError("English-only checkpoints require English transcription")
    if (
        args.task == "translate"
        and config["n_audio_layer"] == 32
        and config["n_text_layer"] == 4
    ):
        raise ValueError("turbo does not support translation")
    cpp_hash = file_digest(args.cpp_model)
    converted_path = args.cpp_model.with_suffix(args.cpp_model.suffix + ".json")
    common_checkpoint_verified = False
    conversion = None
    if converted_path.is_file():
        conversion = json.loads(converted_path.read_text())
        if (
            conversion.get("source_files") != fingerprints
            or conversion.get("output_sha256") != cpp_hash
        ):
            raise ValueError(
                "GGML conversion receipt does not match selected artifacts"
            )
        verify_conversion_assets(conversion, config)
        common_checkpoint_verified = True
    receipt_path = args.worker.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("worker_sha256") != file_digest(args.worker):
        raise ValueError("worker binary fingerprint differs from build receipt")
    if receipt.get("worker_source_sha256") != file_digest(
        Path(__file__).with_name("whisper_cpp_worker.cpp")
    ):
        raise ValueError("worker source changed; rebuild worker")
    for library, digest in receipt["libraries"].items():
        if file_digest(library) != digest:
            raise ValueError("linked library fingerprint changed")
    current = source_identity(receipt["source"]["path"])
    if current["build_inputs_sha256"] != receipt["source"]["build_inputs_sha256"]:
        raise ValueError("whisper.cpp source changed; rebuild worker and library")
    return config, {
        "declared_checkpoint": args.checkpoint,
        "common_checkpoint_verified": common_checkpoint_verified,
        "mlx": fingerprints,
        "whisper_cpp_sha256": cpp_hash,
        "conversion": conversion,
        "worker_build": receipt,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--cpp-model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Declared common checkpoint identity; receipt verifies converted models",
    )
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--references",
        type=Path,
        help="JSON list of human reference texts, one per audio input",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Explicit ISO language; automatic detection is a separate workload",
    )
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--compile-encoder", action="store_true")
    parser.add_argument("--compile-decoder", action="store_true")
    parser.add_argument("--joint-decoding", action="store_true")
    parser.add_argument("--decode-chunk-size", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-conditioning", action="store_true")
    parser.add_argument("--without-timestamps", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--require-uncontended",
        action="store_true",
        help="Fail closed unless bounded OS telemetry windows show low contention; does not guarantee exclusive hardware ownership",
    )
    parser.add_argument("--contention-samples", type=int, default=3)
    parser.add_argument("--contention-interval", type=float, default=0.5)
    parser.add_argument("--contention-cooldown", type=float, default=2)
    parser.add_argument("--max-background-gpu-percent", type=float, default=5)
    parser.add_argument("--max-background-cpu-percent", type=float, default=20)
    parser.add_argument("--max-competing-compute-cpu-percent", type=float, default=5)
    parser.add_argument("--min-memory-free-percent", type=float, default=10)
    parser.add_argument("--max-wer-regression", type=float, default=0)
    parser.add_argument("--max-cer-regression", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    if args.joint_decoding and (
        not args.compile_decoder or args.pipeline or args.beam_size != 1
    ):
        parser.error(
            "joint decoding requires compiled greedy decoding without pipeline"
        )
    if args.decode_chunk_size != 1 and not args.joint_decoding:
        parser.error("decode chunk size requires joint decoding")
    if (
        args.threads < 1
        or args.beam_size < 1
        or args.max_wer_regression < 0
        or args.max_cer_regression < 0
    ):
        parser.error("threads/beam must be positive; quality tolerances nonnegative")
    suite = BenchmarkSuite(
        BenchmarkConfig(
            cases=(BenchmarkCase(label="mlx"), BenchmarkCase(label="whisper_cpp")),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
    )
    config, artifacts = validate_artifacts(args)
    references = json.loads(args.references.read_text()) if args.references else None
    if references is not None and (
        not isinstance(references, list)
        or len(references) != len(args.audio)
        or any(not isinstance(text, str) for text in references)
    ):
        parser.error("references must contain one text string per audio input")
    clips = [decode_pcm(path) for path in args.audio]
    if sum(clip.size for clip in clips) > MAX_SAMPLES:
        parser.error("total canonical audio duration must be <=30 minutes")
    inputs = [
        {
            "path": str(path.resolve()),
            "source_sha256": file_digest(path),
            "pcm_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
            "samples": clip.size,
            "seconds": clip.size / 16000,
        }
        for path, clip in zip(args.audio, clips)
    ]
    options = {
        "language": args.language,
        "task": args.task,
        "temperature": 0.0,
        "temperature_fallback": False,
        "condition_on_previous_text": not args.no_conditioning,
        "segment_timestamps": not args.without_timestamps,
        "word_timestamps": False,
        "beam_size": args.beam_size,
        "best_of": 1,
        "patience": 1,
        "mlx_compile_encoder": args.compile_encoder,
        "mlx_compile_decoder": args.compile_decoder,
        "mlx_pipeline": args.pipeline,
        "mlx_joint_decoding": args.joint_decoding,
        "mlx_decode_chunk_size": args.decode_chunk_size,
        "mlx_constant_pcm_rejection": True,
        "constant_pcm_policy": "MLX rejects exactly constant finite PCM with real centered-STFT context; cpp native policy unchanged; speech-only ratios must be reported separately",
        "length_penalty": None,
        "suppress_blank": True,
        "suppress_non_speech_tokens": True,
        "logprob_threshold": -1,
        "no_speech_threshold": 0.6,
        "compression_fallback": False,
        "vad": False,
        "precision": "f16 weights/MLX activations; cpp native mixed f16/f32",
        "cpp_flash_attention": True,
        "cpp_aneforge": False,
        "cpp_threads": args.threads,
    }
    policy = ContentionPolicy(
        samples=args.contention_samples,
        interval_seconds=args.contention_interval,
        cooldown_seconds=args.contention_cooldown,
        max_gpu_percent=args.max_background_gpu_percent,
        max_cpu_busy_percent=args.max_background_cpu_percent,
        max_compute_cpu_percent=args.max_competing_compute_cpu_percent,
        min_memory_free_percent=args.min_memory_free_percent,
    )
    guard = QuietGuard(policy, required=args.require_uncontended)

    def blocked_by_contention(error):
        summary = guard.summary()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(
                {
                    "status": "blocked-by-contention",
                    "error": str(error),
                    "contention": summary,
                    "options": options,
                    "artifacts": artifacts,
                    "inputs": inputs,
                },
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        return 3 if summary["status"] == "unknown" else 4

    try:
        guard.check("preflight-before-model-loading", cooldown=True)
    except ContentionError as error:
        return blocked_by_contention(error)
    # Import/loading are outside warmed PCM-to-result timing for both engines.
    start = time.perf_counter()
    import mlx.core as mx
    from lightning_whisper_mlx.load_models import load_model
    from lightning_whisper_mlx.transcribe import transcribe_audio

    import_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = load_model(str(args.mlx_model.resolve()), dtype=mx.float16)
    if args.compile_encoder or args.compile_decoder:
        model.configure_execution(
            compile_encoder=args.compile_encoder,
            compile_decoder=args.compile_decoder,
            joint_decoding=args.joint_decoding,
            decode_chunk_size=args.decode_chunk_size,
        )
    mx.eval(model.parameters())
    mx.synchronize()
    mlx_setup = time.perf_counter() - start
    suppression_ids = model_suppression_ids(model, args.language, args.task)
    options["suppression"] = {
        "ids": list(suppression_ids),
        "sha256": hashlib.sha256(
            json.dumps(suppression_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "policy": "canonical MLX -1 plus special tokens and all language tokens; identical explicit IDs; cpp native NST disabled",
        "cpp_native_special_language_timestamp_filters": True,
    }
    start = time.perf_counter()
    with CppBackend(
        args.worker.resolve(),
        args.cpp_model.resolve(),
        language=args.language,
        threads=args.threads,
        beam_size=args.beam_size,
        condition=not args.no_conditioning,
        timestamps=not args.without_timestamps,
        task=args.task,
        timeout=args.timeout,
        suppress_tokens=suppression_ids,
    ) as cpp:
        process = getattr(cpp, "process", None)
        if process is not None:
            guard.owned_pids.add(process.pid)
        cpp_setup = time.perf_counter() - start
        for key, value in cpp.ready["dimensions"].items():
            if config[key] != value:
                raise ValueError("loaded cpp model dimensions differ")
        for index, clip in enumerate(clips):
            cpp.upload(index, clip.tobytes())

        def runner(case):
            outputs = []
            for index, clip in enumerate(clips):
                wall_start = time.perf_counter()
                if case.label == "whisper_cpp":
                    result = cpp.transcribe(index)
                else:
                    started = time.perf_counter()
                    decode_options = {
                        "language": args.language,
                        "task": args.task,
                        "fp16": True,
                        "pipeline": args.pipeline,
                        "without_timestamps": args.without_timestamps,
                        "suppress_tokens": list(suppression_ids),
                        "suppress_blank": True,
                    }
                    if args.beam_size > 1:
                        decode_options.update(beam_size=args.beam_size, patience=1.0)
                    result = transcribe_audio(
                        clip,
                        path_or_hf_repo=str(args.mlx_model.resolve()),
                        model_instance=model,
                        temperature=0.0,
                        compression_ratio_threshold=None,
                        logprob_threshold=-1,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=not args.no_conditioning,
                        word_timestamps=False,
                        batch_size=1,
                        verbose=None,
                        **decode_options,
                    )
                    mx.synchronize()
                    result = {
                        "text": result["text"],
                        "language": result["language"],
                        "segments": result["segments"],
                        "tokens": [
                            token
                            for segment in result["segments"]
                            for token in segment["tokens"]
                        ],
                        "service_seconds": time.perf_counter() - started,
                    }
                result["wall_seconds"] = time.perf_counter() - wall_start
                outputs.append(result)
            return outputs

        completed_samples = 0
        pair_count = len(suite.config.cases)

        def before_sample():
            if completed_samples % pair_count == 0:
                guard.check(
                    f"before-timed-pair-{completed_samples // pair_count}",
                    cooldown=True,
                )

        def after_sample():
            nonlocal completed_samples
            completed_samples += 1
            status = None
            if completed_samples % pair_count == 0:
                status = guard.check(
                    f"after-timed-pair-{completed_samples // pair_count - 1}",
                    cooldown=True,
                )["status"]
            return {
                "paired_round": (completed_samples - 1) // pair_count,
                "contention_after_pair": status,
            }

        try:
            results = suite.run(
                runner,
                before_sample=before_sample,
                after_sample=after_sample,
                metadata={
                    "artifacts": artifacts,
                    "inputs": inputs,
                    "options": options,
                    "cpp_runtime": cpp.ready,
                    "timing_scope": "warm resident canonical PCM to structured result; cpp request/response IPC included; upload excluded",
                    "cold_setup_seconds": {
                        "mlx_import": import_seconds,
                        "mlx_model": mlx_setup,
                        "cpp_process_and_model": cpp_setup,
                    },
                    "cold_scope": "in-process setup diagnostics; not cold CLI latency",
                    "reference_sha256": (
                        file_digest(args.references) if args.references else None
                    ),
                },
            )
        except ContentionError as error:
            return blocked_by_contention(error)
    assessment = assess_backend_results(
        results,
        references,
        max_wer_regression=args.max_wer_regression,
        max_cer_regression=args.max_cer_regression,
    )
    if not artifacts["common_checkpoint_verified"]:
        assessment["speed_claim_qualified"] = False
        assessment["checkpoint_status"] = (
            "declaration-only; provide conversion receipt for byte-verified shared weights"
        )
    if not args.without_timestamps:
        assessment["speed_claim_qualified"] = False
    contention = guard.summary()
    assessment["contention_qualified"] = contention["qualified"]
    if not contention["qualified"]:
        assessment["speed_claim_qualified"] = False
    report = {
        **suite.to_dict(results),
        "assessment": assessment,
        "contention": contention,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
