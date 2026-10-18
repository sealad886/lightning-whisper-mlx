"""Compare interleaved policies and separately profile synchronized inference stages."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from .runner import BenchmarkCase, BenchmarkConfig, BenchmarkSuite, environment_metadata

POLICIES = (
    ("eager", False, False, False),
    ("eager_pipeline", False, False, True),
    ("compiled_encoder", True, False, False),
    ("compiled_decoder", False, True, False),
    ("compiled_both", True, True, False),
    ("compiled_both_pipeline", True, True, True),
)


def compare_policies(
    runner,
    *,
    repetitions=7,
    warmups=2,
    before_sample=None,
    after_sample=None,
    metadata=None
):
    """Compare the bounded policy matrix with rotating, interleaved samples.

    The injected runner receives (name, compile_encoder, compile_decoder,
    pipeline), synchronizes pending work, and returns its observable result.
    Model loading and creation of compiled callables happen before this helper.
    """
    policies = {policy[0]: policy for policy in POLICIES}
    suite = BenchmarkSuite(
        BenchmarkConfig(
            cases=tuple(BenchmarkCase(label=name) for name in policies),
            warmups=warmups,
            repetitions=repetitions,
        )
    )
    results = suite.run(
        lambda case: runner(policies[case.label]),
        before_sample=before_sample,
        after_sample=after_sample,
        metadata=metadata,
    )
    payload = suite.to_dict(results)
    baseline = results[0].median_seconds
    payload["policies"] = [
        {
            "name": name,
            "compile_encoder": encoder,
            "compile_decoder": decoder,
            "pipeline": pipeline,
        }
        for name, encoder, decoder, pipeline in POLICIES
    ]
    payload["relative_to_eager"] = {
        result.case.label: baseline / result.median_seconds for result in results
    }
    outputs = [output for result in results for output in result.outputs]
    payload["text_outputs_equal"] = all(
        output.get("text") == outputs[0].get("text") for output in outputs
    )
    payload["scope"] = (
        "interleaved warmed PCM-to-result policy comparison; compiled callable "
        "creation and model loading excluded; synchronize before runner returns"
    )
    return payload


def profile(
    model,
    pcm,
    *,
    repetitions=7,
    warmups=2,
    language="en",
    compile_encoder=False,
    compile_decoder=False,
    pipeline=False,
    word_timestamps=False
):
    import mlx.core as mx
    from lightning_whisper_mlx.transcribe import transcribe_audio

    model.configure_execution(
        compile_encoder=compile_encoder, compile_decoder=compile_decoder
    )
    options = dict(
        model_instance=model,
        language=language,
        temperature=0.0,
        word_timestamps=word_timestamps,
        pipeline=pipeline,
        batch_size=1,
    )
    durations = []
    outputs = []
    for index in range(warmups + repetitions):
        if index == warmups:
            mx.synchronize()
            mx.reset_peak_memory()
        mx.synchronize()
        start = time.perf_counter()
        result = transcribe_audio(pcm, **options)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        if index >= warmups:
            durations.append(elapsed)
            outputs.append(result)
    memory = {
        "active_bytes": mx.get_active_memory(),
        "peak_active_bytes": mx.get_peak_memory(),
        "cached_bytes": mx.get_cache_memory(),
        "scope": (
            "shared-process allocation after this policy's sequential samples; "
            "includes resident model and other policy allocations; peak reset "
            "before first timed sample; compilation included when warmups=0; "
            "subsequent instrumented invocation excluded"
        ),
    }

    # Stage synchronization changes scheduling. These samples are diagnostics,
    # never substituted for the uninstrumented wall times measured above.
    stages = {}
    encode, decoder = model.embed_audio, model.decoder
    compiled_step = getattr(model, "_compiled_decoder", None)
    from lightning_whisper_mlx import transcribe as transcription

    prepare_audio = transcription.log_mel_spectrogram

    def timed(name, function, *args, **kwargs):
        mx.synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        mx.eval(result[0] if isinstance(result, tuple) else result)
        stages.setdefault(name, []).append(time.perf_counter() - start)
        return result

    def timed_decoder(*args, **kwargs):
        name = (
            "alignment"
            if kwargs.get("return_scores")
            else "prefill" if kwargs.get("kv_cache") is None else "cached_step"
        )
        return timed(name, decoder, *args, **kwargs)

    model.embed_audio = lambda *args, **kwargs: timed(
        "encoder", encode, *args, **kwargs
    )
    model.decoder = timed_decoder
    if compiled_step is not None:
        model._compiled_decoder = lambda *args: timed(
            "cached_step", compiled_step, *args
        )
    transcription.log_mel_spectrogram = lambda *args, **kwargs: timed(
        "audio_preparation", prepare_audio, *args, **kwargs
    )
    instrumented_start = time.perf_counter()
    try:
        transcribe_audio(pcm, **options)
        mx.synchronize()
    finally:
        model.embed_audio, model.decoder = encode, decoder
        model._compiled_decoder = compiled_step
        transcription.log_mel_spectrogram = prepare_audio
    instrumented_seconds = time.perf_counter() - instrumented_start
    residual = max(0.0, instrumented_seconds - sum(sum(v) for v in stages.values()))
    stages["host_orchestration_and_postprocessing"] = [residual]
    return {
        "policy": {
            "compile_encoder": compile_encoder,
            "pipeline": pipeline,
            "compile_decoder": compile_decoder,
            "word_timestamps": word_timestamps,
        },
        "durations_seconds": durations,
        "median_seconds": statistics.median(durations),
        "latency_scope": (
            "sequential per-policy diagnostics; use policy_comparison for selection"
        ),
        "outputs": outputs,
        "stages": {
            key: {
                "count": len(values),
                "total_seconds": sum(values),
                "median_seconds": statistics.median(values),
            }
            for key, values in stages.items()
        },
        "stage_scope": "one separately instrumented invocation; synchronization perturbs scheduling",
        "memory": memory,
    }


def main(argv=None):
    from .compare_backends import decode_pcm
    from .cpp_backend import file_digest
    import mlx.core as mx
    from lightning_whisper_mlx.load_models import load_model
    from lightning_whisper_mlx.transcribe import transcribe_audio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.repetitions < 1 or args.warmups < 0:
        parser.error("fresh output and positive repetition count required")
    model = load_model(str(args.model), dtype=mx.float16, local_files_only=True)
    pcm = decode_pcm(args.audio)
    input_metadata = {
        "path": str(args.audio.resolve()),
        "source_sha256": file_digest(args.audio),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "samples": int(pcm.size),
        "sample_rate": 16000,
    }
    execution = {}
    for name, encoder, decoder, _ in POLICIES:
        model.configure_execution(compile_encoder=encoder, compile_decoder=decoder)
        execution[name] = (model._compiled_encoder, model._compiled_decoder)

    def run_policy(policy):
        name, _, _, pipeline = policy
        model._compiled_encoder, model._compiled_decoder = execution[name]
        result = transcribe_audio(
            pcm,
            model_instance=model,
            language=args.language,
            temperature=0.0,
            word_timestamps=args.word_timestamps,
            pipeline=pipeline,
            batch_size=1,
        )
        mx.synchronize()
        return result

    active_before = 0

    def before_sample():
        nonlocal active_before
        mx.synchronize()
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()

    def after_sample():
        return {
            "active_bytes_before": active_before,
            "active_bytes_after": mx.get_active_memory(),
            "peak_active_bytes": mx.get_peak_memory(),
            "peak_increment_bytes": max(0, mx.get_peak_memory() - active_before),
            "cached_bytes": mx.get_cache_memory(),
            "scope": (
                "shared-process allocation for this interleaved timed sample; "
                "all policy callables and model resident; peak reset immediately "
                "before sample; allocator cache retained; not isolated policy residency"
            ),
        }

    comparison = compare_policies(
        run_policy,
        repetitions=args.repetitions,
        warmups=args.warmups,
        before_sample=before_sample,
        after_sample=after_sample,
        metadata={
            "model": model._checkpoint_fingerprint,
            "input": input_metadata,
            "precision": "float16",
            "language": args.language,
            "word_timestamps": args.word_timestamps,
        },
    )
    reports = []
    for _, compile_encoder, compile_decoder, pipeline in POLICIES:
        reports.append(
            profile(
                model,
                pcm,
                repetitions=args.repetitions,
                warmups=args.warmups,
                language=args.language,
                compile_encoder=compile_encoder,
                compile_decoder=compile_decoder,
                pipeline=pipeline,
                word_timestamps=args.word_timestamps,
            )
        )
    payload = {
        "environment": environment_metadata(),
        "model": model._checkpoint_fingerprint,
        "input": input_metadata,
        "policy_comparison": comparison,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, default=lambda value: float(value), allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
