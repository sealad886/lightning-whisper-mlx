"""Packed attention preserves source checkpoint projection and cache contracts."""

from dataclasses import asdict
import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from lightning_whisper_mlx.load_models import load_model, pack_attention_weights
from lightning_whisper_mlx.whisper import ModelDimensions, MultiHeadAttention, Whisper


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("cross", [False, True])
def test_source_attention_projections_pack_without_requantization(quantized, cross):
    source = {
        name: nn.Linear(32, 32, bias=name != "key")
        for name in ("query", "key", "value")
    }
    if quantized:
        source = {
            name: layer.to_quantized(group_size=32, bits=4)
            for name, layer in source.items()
        }
    prefix = "layer.cross_attn" if cross else "layer.attn"
    weights = {
        f"{prefix}.{name}.{key}": value
        for name, layer in source.items()
        for key, value in tree_flatten(layer.parameters())
    }
    packed = pack_attention_weights(weights)
    attention = MultiHeadAttention(32, 4, cross_attention=cross)
    if quantized:
        nn.quantize(
            attention,
            group_size=32,
            bits=4,
            class_predicate=lambda path, layer: f"{prefix}.{path}.scales" in packed,
        )
    attention.update(
        tree_unflatten(
            [(name.removeprefix(prefix + "."), value) for name, value in packed.items()]
        )
    )
    x, audio = mx.random.normal((2, 3, 32)), mx.random.normal((2, 7, 32))
    if cross:
        actual = [attention.query(x), *mx.split(attention.kv(audio), 2, axis=-1)]
        expected = [source["query"](x), source["key"](audio), source["value"](audio)]
    else:
        actual = mx.split(attention.qkv(x), 3, axis=-1)
        expected = [source[name](x) for name in ("query", "key", "value")]
    assert all(
        mx.allclose(a, b, atol=1e-4, rtol=1e-4).item() for a, b in zip(actual, expected)
    )
    assert not any(
        name.endswith(".key.weight") or name.endswith(".value.weight")
        for name in packed
    )
    if quantized:
        name = "kv" if cross else "qkv"
        assert packed[f"{prefix}.{name}.weight"].dtype == mx.uint32
        assert packed[f"{prefix}.{name}.scales"].shape[0] == (64 if cross else 96)


def _separate_checkpoint(weights):
    output = dict(weights)
    for name in list(weights):
        if ".qkv." in name:
            prefix, suffix = name.split(".qkv.")
            sources = ("query", "key", "value")
        elif ".kv." in name:
            prefix, suffix = name.split(".kv.")
            sources = ("key", "value")
        else:
            continue
        output.pop(name)
        for source, value in zip(
            sources, mx.split(weights[name], len(sources), axis=0)
        ):
            if not (source == "key" and suffix == "bias"):
                output[f"{prefix}.{source}.{suffix}"] = value
    return output


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_source_and_canonical_checkpoint_loads_preserve_model_outputs(
    tmp_path, quantized, separate
):
    model = Whisper(ModelDimensions(4, 3, 32, 2, 1, 32, 16, 32, 2, 1), mx.float32)
    config = asdict(model.dims)
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    weights = dict(tree_flatten(model.parameters()))
    if separate:
        weights = _separate_checkpoint(weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.savez(str(tmp_path / "weights.npz"), **weights)
    loaded = load_model(tmp_path)
    features, tokens = mx.random.normal((1, 3, 32)), mx.array([[1, 2, 3]])
    assert mx.allclose(
        loaded.logits(tokens, features),
        model.logits(tokens, features),
        atol=2e-4,
        rtol=2e-4,
    ).item()
    original_keys = set(weights)
    assert set(mx.load(str(tmp_path / "weights.npz"))) == original_keys
    loaded_keys = dict(tree_flatten(loaded.parameters()))
    assert any(".qkv.weight" in name for name in loaded_keys)
    assert not any(
        ".key.weight" in name or ".value.weight" in name for name in loaded_keys
    )


def test_cross_attention_reuses_packed_audio_cache():
    attention = MultiHeadAttention(16, 4, cross_attention=True)
    x, audio = mx.random.normal((1, 2, 16)), mx.random.normal((1, 5, 16))
    expected, cache, _ = attention(x, audio)
    actual, reused, _ = attention(x, None, kv_cache=cache)
    assert mx.allclose(actual, expected, atol=1e-5).item()
    assert reused[0] is cache[0] and reused[1] is cache[1]


@pytest.mark.parametrize("problem", ["missing", "mixed", "shape", "both"])
def test_inconsistent_source_projection_groups_fail_clearly(problem):
    weights = {
        f"layer.attn.{name}.weight": mx.zeros((32, 32))
        for name in ("query", "key", "value")
    }
    if problem == "missing":
        weights.pop("layer.attn.key.weight")
    elif problem == "mixed":
        weights["layer.attn.query.scales"] = mx.ones((32, 1))
    elif problem == "shape":
        weights["layer.attn.key.weight"] = mx.zeros((16, 32))
    else:
        weights["layer.attn.qkv.weight"] = mx.zeros((96, 32))
    with pytest.raises(ValueError, match="projection"):
        pack_attention_weights(weights)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype,tolerance", [(mx.float32, 2e-5), (mx.float16, 0.004)])
def test_functional_cached_step_matches_owner_cache_and_returns_explicit_state(
    compiled, dtype, tolerance
):
    model = Whisper(ModelDimensions(4, 3, 32, 2, 1, 32, 16, 32, 2, 1), dtype)
    # Explicit conversion is required for a freshly constructed test model too.
    from mlx.utils import tree_map

    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    features = mx.random.normal((2, 3, 32)).astype(dtype)
    prefix = mx.array([[1, 2, 3], [3, 2, 1]])
    _, owners, _, _ = model.decoder(prefix, features, return_hidden=True)
    _, independent, _, _ = model.decoder(prefix, features, return_hidden=True)
    state = [(owner.state, cross) for owner, cross in independent]
    model.configure_execution(compile_decoder=compiled)
    for offset, token in enumerate((4, 5, 6), start=3):
        tokens = mx.array([[token], [token + 1]])
        expected, owners, _, expected_hidden = model.decoder(
            tokens, features, kv_cache=owners, return_hidden=True
        )
        if compiled:
            actual, state, hidden = model._compiled_decoder(
                tokens, features, state, mx.array(offset)
            )
        else:
            actual, state, _, hidden = model.decoder(
                tokens,
                features,
                kv_cache=state,
                offset=mx.array(offset),
                return_hidden=True,
            )
        assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()
        assert mx.allclose(
            hidden, expected_hidden, atol=tolerance, rtol=tolerance
        ).item()
        for (keys, _), ((actual_keys, actual_values), _) in zip(owners, state):
            assert mx.allclose(
                actual_keys[:, : offset + 1],
                keys.keys[:, : offset + 1],
                atol=tolerance,
                rtol=tolerance,
            ).item()
            assert mx.allclose(
                actual_values[:, : offset + 1],
                keys.values[:, : offset + 1],
                atol=tolerance,
                rtol=tolerance,
            ).item()
        assert independent[0][0].offset == 3
        mx.eval(actual, hidden, state)


def test_functional_decoder_requires_one_token_and_existing_buffers():
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 32, 16, 8, 2, 1), mx.float32)
    features = mx.zeros((1, 3, 8))
    with pytest.raises(ValueError, match="one token"):
        model.decoder(mx.array([[1, 2]]), features, offset=mx.array(0))
