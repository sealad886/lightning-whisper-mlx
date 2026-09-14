import mlx.core as mx
import pytest

from lightning_whisper_mlx.decoding import DecodingOptions, GreedyDecoder, decode
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def test_temperature_zero_selects_target_argmax():
    decoder = GreedyDecoder(temperature=0.0, eot=99)
    tokens, completed, logprobs = decoder.update(
        mx.array([[1, 2]]),
        mx.array([[0.1, 0.2, 4.0]]),
        mx.array([0.0]),
    )

    assert tokens.tolist() == [[1, 2, 2]]
    assert not completed.item()
    assert logprobs.shape == (1,)


def test_nonzero_temperature_uses_sampler(monkeypatch):
    sampled = mx.array([1])

    def fake_categorical(*, logits):
        assert logits.shape == (1, 3)
        return sampled

    monkeypatch.setattr(
        "lightning_whisper_mlx.decoding.mx.random.categorical", fake_categorical
    )

    decoder = GreedyDecoder(temperature=1.0, eot=99)
    tokens, completed, _ = decoder.update(
        mx.array([[1, 2]]),
        mx.array([[0.1, 0.2, 4.0]]),
        mx.array([0.0]),
    )

    assert tokens.tolist() == [[1, 2, 1]]
    assert not completed.item()


@pytest.mark.parametrize("fp16", [False, True])
def test_assistant_decode_uses_requested_precision_for_both_encoders(monkeypatch, fp16):
    dtype = mx.float16 if fp16 else mx.float32
    dims = ModelDimensions(
        n_mels=4,
        n_audio_ctx=3,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=51865,
        n_text_ctx=16,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=1,
    )
    target = Whisper(dims, dtype=dtype)
    assistant = Whisper(dims, dtype=dtype)
    encoder_dtypes = {}

    def observe_encoder(model, name):
        model.set_dtype(dtype)
        encoder = model.encoder

        def encode(mel):
            encoder_dtypes[name] = mel.dtype
            return encoder(mel)

        monkeypatch.setattr(model, "encoder", encode)

    observe_encoder(target, "target")
    observe_encoder(assistant, "assistant")
    result = decode(
        target,
        mx.zeros((6, 4), dtype=mx.float32),
        DecodingOptions(
            language="en",
            fp16=fp16,
            without_timestamps=True,
            sample_len=2,
            speculative_mode="assistant",
            assistant_model=assistant,
            lookahead=2,
        ),
    )

    assert encoder_dtypes == {"target": dtype, "assistant": dtype}
    assert result.audio_features.dtype == dtype
    assert len(result.tokens) <= 2
