"""Stage profiling remains outside joint graphs even after warmed keys are evicted."""

import mlx.core as mx
import pytest

from benchmarks.profile_inference import profile
from lightning_whisper_mlx import transcribe
from lightning_whisper_mlx.decoding import GreedyDecoder, Inference, SuppressTokens
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def test_profile_times_new_joint_graph_after_warmed_cache_eviction(monkeypatch):
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 32, 16, 8, 2, 1), mx.float32)
    original_decoder = model.decoder
    original_sample_next = Inference.sample_next
    calls = 0

    def recording(pcm, **options):
        nonlocal calls
        calls += 1
        if calls == 3:
            # The instrumented replay starts with an evicted key, as a recording
            # with many conditioned prompt lengths can do with the bounded cache.
            model._joint_decoder_cache.clear()
        inference = Inference(model, 1)
        sampler = GreedyDecoder(0, 31, 17)
        sampler.configure_sampling([SuppressTokens([31], 32)])
        features, prefix = mx.zeros((1, 3, 8)), mx.array([[1]])
        tokens, _, scores = sampler.update(
            prefix, inference.prefill(prefix, features)[:, -1], mx.zeros(1)
        )
        tokens, _, scores = inference.sample_next(
            tokens, features, sampler, scores, max_steps=4
        )
        mx.eval(tokens, scores, inference.cache_state)
        return {"text": "same", "tokens": tokens.tolist()}

    monkeypatch.setattr(transcribe, "transcribe_audio", recording)
    result = profile(
        model,
        mx.zeros(1),
        repetitions=1,
        warmups=1,
        compile_decoder=True,
        joint_decoding=True,
        decode_chunk_size=4,
    )
    assert calls == 3
    assert result["stages"]["cached_chunk"]["count"] == 1
    assert len(result["durations_seconds"]) == 1
    assert Inference.sample_next is original_sample_next
    assert model.decoder is original_decoder


def test_profile_restores_outer_wrappers_when_instrumented_recording_fails(monkeypatch):
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 32, 16, 8, 2, 1), mx.float32)
    decoder, encode = model.decoder, model.embed_audio
    sample_next, prepare = (
        Inference.sample_next,
        transcribe.log_mel_spectrogram_from_prepared,
    )
    calls = 0

    def recording(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            assert Inference.sample_next is not sample_next
            raise RuntimeError("instrumented request failed")
        return {"text": "same"}

    monkeypatch.setattr(transcribe, "transcribe_audio", recording)
    with pytest.raises(RuntimeError, match="instrumented request failed"):
        profile(
            model,
            mx.zeros(1),
            warmups=1,
            repetitions=1,
            compile_decoder=True,
            joint_decoding=True,
            decode_chunk_size=4,
        )
    assert Inference.sample_next is sample_next
    assert model.decoder is decoder
    assert model.embed_audio == encode
    assert transcribe.log_mel_spectrogram_from_prepared is prepare
