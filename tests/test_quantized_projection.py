import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from lightning_whisper_mlx.load_models import load_model
from lightning_whisper_mlx.mtp import MTPHead
from lightning_whisper_mlx.speculative import MTPSpeculativeDecoder
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def embedding_and_dense_weight(quantized):
    embedding = nn.Embedding(32, 32)
    if not quantized:
        return embedding, embedding.weight
    embedding = embedding.to_quantized(group_size=32, bits=4)
    weight = mx.dequantize(
        embedding.weight,
        embedding.scales,
        embedding.biases,
        group_size=32,
        bits=4,
    )
    return embedding, weight


@pytest.mark.parametrize("quantized", [False, True])
def test_mtp_head_uses_tied_vocabulary_projection(quantized):
    embedding, weight = embedding_and_dense_weight(quantized)
    head = MTPHead(hidden_size=32, horizons=2)
    # A zero residual head must use the original hidden state at each horizon.
    for projection in head.projections:
        projection.weight = mx.zeros_like(projection.weight)
        projection.bias = mx.zeros_like(projection.bias)
    hidden = mx.random.normal((1, 2, 32))

    logits = head(hidden, embedding.as_linear)

    expected = mx.stack([hidden @ weight.T] * 2, axis=-2)
    assert logits.shape == (1, 2, 2, 32)
    assert mx.allclose(logits, expected, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("quantized", [False, True])
def test_loaded_mtp_decode_matches_greedy_with_tied_projection(tmp_path, quantized):
    model = Whisper(
        ModelDimensions(
            n_mels=4,
            n_audio_ctx=3,
            n_audio_state=32,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=32,
            n_text_ctx=16,
            n_text_state=32,
            n_text_head=2,
            n_text_layer=1,
        ),
        dtype=mx.float32,
    )
    model.decoder.token_embedding, weight = embedding_and_dense_weight(quantized)
    config = asdict(model.dims)
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.savez(str(tmp_path / "weights.npz"), **dict(tree_flatten(model.parameters())))
    model = load_model(str(tmp_path))

    audio = mx.random.normal((1, 3, 32))
    prefix = [1, 2]
    logits, hidden = model.logits_and_hidden(mx.array([prefix]), audio)
    assert logits.shape == (1, 2, 32)
    assert mx.allclose(logits, hidden @ weight.T, atol=1e-4, rtol=1e-4).item()

    history = list(prefix)
    expected_logprob = 0.0
    for _ in range(4):
        row = model.logits(mx.array([history]), audio)[:, -1]
        token = int(mx.argmax(row, axis=-1)[0].item())
        expected_logprob += float(
            (row - mx.logsumexp(row, axis=-1, keepdims=True))[0, token].item()
        )
        history.append(token)

    decoder = MTPSpeculativeDecoder(
        model=model,
        head=MTPHead(hidden_size=32, horizons=2),
        audio_features=audio,
        prefix=prefix,
        lookahead=3,
        eot=-1,
    )
    actual, actual_logprob = decoder.decode(sample_len=4)

    assert actual == history[len(prefix) :]
    assert actual_logprob == pytest.approx(expected_logprob, abs=1e-4)
