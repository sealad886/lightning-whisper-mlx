"""Invalid generation limits fail before audio encoding or decoder execution."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from lightning_whisper_mlx.decoding import DecodingOptions, DecodingTask, decode


@pytest.fixture
def model_without_inference():
    def unexpected_encoding(*args, **kwargs):
        pytest.fail("invalid options reached audio encoding")

    return SimpleNamespace(
        dims=SimpleNamespace(
            n_text_ctx=32, n_vocab=51865, n_audio_ctx=3, n_audio_state=8
        ),
        is_multilingual=True,
        num_languages=99,
        embed_audio=unexpected_encoding,
    )


@pytest.mark.parametrize("entrypoint", ["task", "decode"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_len", False),
        ("sample_len", True),
        ("sample_len", 0),
        ("sample_len", -1),
        ("sample_len", 1.5),
        ("temperature", -1.0),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", -float("inf")),
    ],
)
def test_invalid_options_fail_before_inference(
    model_without_inference, entrypoint, field, value
):
    options = DecodingOptions(language="en", **{field: value})
    with pytest.raises(ValueError, match=field):
        if entrypoint == "task":
            DecodingTask(model_without_inference, options)
        else:
            decode(model_without_inference, mx.zeros((6, 4)), options)


@pytest.mark.parametrize("temperature", [0, 0.25, 1.0])
@pytest.mark.parametrize("sample_len", [None, 1, 64])
def test_valid_options_keep_generation_upper_limit(
    model_without_inference, temperature, sample_len
):
    options = DecodingOptions(
        language="en", temperature=temperature, sample_len=sample_len
    )
    task = DecodingTask(model_without_inference, options)
    assert task.options.temperature == temperature
    assert task.sample_len == (16 if sample_len is None else sample_len)
