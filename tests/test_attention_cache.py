"""Cache ownership, context bounds, and fused numerical attention contracts."""

import mlx.core as mx
import pytest

from lightning_whisper_mlx.cache import KVCache
from lightning_whisper_mlx.whisper import MultiHeadAttention
from test_block_inference import make_model


def test_cache_append_truncate_replay_and_context_bound():
    cache = KVCache(4)
    keys = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    cache.update(keys, keys + 1)
    cache.update(mx.ones((1, 1, 2)), mx.ones((1, 1, 2)))
    cache.truncate(2)
    actual, values = cache.update(mx.full((1, 2, 2), 9.0), mx.full((1, 2, 2), 8.0))
    assert actual.tolist() == [[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0], [9.0, 9.0]]]
    assert values[:, :2].tolist() == (keys + 1).tolist()
    assert cache.offset == cache.max_size == 4
    with pytest.raises(ValueError, match="context"):
        cache.update(mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)))
    with pytest.raises(ValueError, match="truncation"):
        cache.truncate(5)


def test_beam_cache_reordering_duplicates_only_selected_audio_rows():
    cache = KVCache(4)
    keys = mx.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    cache.update(keys, keys + 1)
    cache.reorder([1, 1, 0])
    assert next(iter(cache)).tolist() == [[[3.0, 4.0]], [[3.0, 4.0]], [[1.0, 2.0]]]
    assert cache.offset == 1


@pytest.mark.parametrize("dtype,tolerance", [(mx.float32, 1e-5), (mx.float16, 0.004)])
@pytest.mark.parametrize("length", [1, 4])
def test_fused_attention_matches_explicit_reference(dtype, tolerance, length):
    mx.random.seed(4)
    attention = MultiHeadAttention(16, 4)
    q = mx.random.normal((2, length, 16)).astype(dtype)
    k = mx.random.normal((2, 7, 16)).astype(dtype)
    v = mx.random.normal((2, 7, 16)).astype(dtype)
    actual, scores = attention.qkv_attention(q, k, v)
    expected, reference_scores = attention.qkv_attention(q, k, v, return_scores=True)
    assert scores is None and reference_scores.shape == (2, 4, length, 7)
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()


def test_encoder_compilation_matches_eager_encoder():
    model = make_model()
    mel = mx.random.normal((1, 6, 4))
    expected = model.embed_audio(mel)
    model.configure_execution(compile_encoder=True)
    actual = model.embed_audio(mel)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
