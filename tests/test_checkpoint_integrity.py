"""Inference rejects incomplete learned tensors and validates alignment metadata."""

from dataclasses import asdict
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from lightning_whisper_mlx.load_models import load_model
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def checkpoint(path, *, quantized=False, filename="weights.npz", mutate=None):
    model = Whisper(ModelDimensions(4, 3, 32, 2, 1, 32, 16, 32, 2, 2), mx.float32)
    config = asdict(model.dims)
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    weights = dict(tree_flatten(model.parameters()))
    if mutate is not None:
        mutate(weights, config)
    (path / "config.json").write_text(json.dumps(config))
    if filename.endswith("safetensors"):
        mx.save_safetensors(str(path / filename), weights)
    else:
        mx.savez(str(path / filename), **weights)
    return model


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("filename", ["weights.npz", "weights.safetensors"])
@pytest.mark.parametrize("defect", ["missing", "extra", "shape"])
def test_checkpoint_rejects_invalid_learned_tensor_set(
    tmp_path, quantized, filename, defect
):
    def damage(weights, config):
        if defect == "missing":
            weights.pop("decoder.ln.weight")
        elif defect == "extra":
            weights["unknown.weight"] = mx.zeros((32,))
        else:
            weights["decoder.ln.weight"] = mx.zeros((33,))

    checkpoint(tmp_path, quantized=quantized, filename=filename, mutate=damage)
    with pytest.raises(ValueError):
        load_model(tmp_path)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "prefix,missing_bias",
    [
        ("decoder.blocks.0.attn", "query"),
        ("decoder.blocks.0.attn", "value"),
        ("decoder.blocks.0.cross_attn", "value"),
    ],
)
def test_separate_projection_requires_each_learned_bias(
    tmp_path, quantized, prefix, missing_bias
):
    def damage(weights, config):
        packed = "kv" if prefix.endswith("cross_attn") else "qkv"
        sources = ("key", "value") if packed == "kv" else ("query", "key", "value")
        for suffix in ("weight", "bias", "scales", "biases"):
            value = weights.pop(f"{prefix}.{packed}.{suffix}", None)
            if value is None:
                continue
            for name, part in zip(sources, mx.split(value, len(sources), axis=0)):
                if suffix == "bias" and name in ("key", missing_bias):
                    continue
                weights[f"{prefix}.{name}.{suffix}"] = part

    checkpoint(tmp_path, quantized=quantized, mutate=damage)
    with pytest.raises(ValueError, match="bias"):
        load_model(tmp_path)


@pytest.mark.parametrize("saved_heads", [None, [[0, 1]]])
def test_optional_checkpoint_alignment_heads_roundtrip(tmp_path, saved_heads):
    def metadata(weights, config):
        if saved_heads is None:
            weights.pop("alignment_heads")
        else:
            weights["alignment_heads"] = mx.array(saved_heads)

    original = checkpoint(tmp_path, mutate=metadata)
    loaded = load_model(tmp_path)
    expected = original.alignment_heads if saved_heads is None else saved_heads
    assert np.array_equal(np.asarray(loaded.alignment_heads), np.asarray(expected))


@pytest.mark.parametrize("saved_heads", [[], [[0, 2]], [[0.5, 1.0]]])
def test_checkpoint_alignment_heads_must_be_valid_pairs(tmp_path, saved_heads):
    checkpoint(
        tmp_path,
        mutate=lambda weights, config: weights.update(
            alignment_heads=mx.array(saved_heads)
        ),
    )
    with pytest.raises(ValueError, match="alignment_heads"):
        load_model(tmp_path)


def test_config_alignment_heads_override_checkpoint_metadata(tmp_path):
    def metadata(weights, config):
        weights["alignment_heads"] = mx.array([[0, 1]])
        config["alignment_heads"] = [[1, 0], [1, 1]]

    checkpoint(tmp_path, mutate=metadata)
    loaded = load_model(tmp_path)
    assert np.array_equal(np.asarray(loaded.alignment_heads), [[1, 0], [1, 1]])
