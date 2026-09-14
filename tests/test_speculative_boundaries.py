import mlx.core as mx
import pytest

from lightning_whisper_mlx.mtp import MTPHead
from lightning_whisper_mlx.speculative import (
    AssistantSpeculativeDecoder,
    MTPSpeculativeDecoder,
)
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


@pytest.mark.parametrize("mode", ["assistant", "mtp"])
@pytest.mark.parametrize(
    "prefix_length,sample_len", [(15, 1), (15, 6), (16, 3), (13, 2)]
)
@pytest.mark.parametrize("stop_at_eot", [False, True])
def test_speculative_decode_respects_context_and_sample_budget(
    mode, prefix_length, sample_len, stop_at_eot
):
    model = Whisper(
        ModelDimensions(
            n_mels=4,
            n_audio_ctx=3,
            n_audio_state=8,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=32,
            n_text_ctx=16,
            n_text_state=8,
            n_text_head=2,
            n_text_layer=1,
        ),
        dtype=mx.float32,
    )
    audio = mx.random.normal((1, 3, 8))
    prefix = [1, 2, 3, 4] * 4
    prefix = prefix[:prefix_length]
    first_logits = model.logits(mx.array([prefix]), audio)[:, -1]
    eot = int(mx.argmax(first_logits, axis=-1)[0].item()) if stop_at_eot else -1

    # Baseline can emit one final token from a full context, without replay.
    history = list(prefix)
    expected = []
    expected_logprob = 0.0
    for _ in range(sample_len):
        logits = model.logits(mx.array([history]), audio)[:, -1].astype(mx.float32)
        token = int(mx.argmax(logits, axis=-1)[0].item())
        expected.append(token)
        expected_logprob += float(
            (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0, token].item()
        )
        history.append(token)
        if token == eot or len(history) > model.dims.n_text_ctx:
            break

    common = dict(audio_features=audio, prefix=prefix, lookahead=3, eot=eot)
    if mode == "assistant":
        decoder = AssistantSpeculativeDecoder(
            target_model=model, assistant_model=model, **common
        )
    else:
        decoder = MTPSpeculativeDecoder(
            model=model, head=MTPHead(hidden_size=8, horizons=2), **common
        )

    actual, actual_logprob = decoder.decode(sample_len=sample_len)

    assert actual == expected
    assert actual_logprob == pytest.approx(expected_logprob, abs=1e-4)
    assert decoder.verifier.cache_length <= model.dims.n_text_ctx
    if mode == "assistant":
        assert decoder.proposer.cache_length <= model.dims.n_text_ctx
