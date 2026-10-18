# Copyright © 2023 Apple Inc.

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper
from .model_catalog import model_spec


def weight_file(model_path):
    """Select the supported root-level weight artifact deterministically."""
    model_path = Path(model_path)
    for name in ("weights.safetensors", "weights.npz"):
        if (model_path / name).is_file():
            return model_path / name
    raise FileNotFoundError(f"{model_path} requires weights.safetensors or weights.npz")


def validate_model_path(model_path):
    model_path = Path(model_path).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model directory requires config.json: {model_path}")
    weight_file(model_path)
    return model_path.resolve()


def resolve_model_path(
    path_or_hf_repo, *, quant=None, local_files_only=False, cache_dir=None
):
    """Resolve a catalog name, local directory, or Hub ID to a validated directory.

    Existing directories are used in place. Explicit missing local paths never
    trigger network access. Hub downloads use its managed cache rather than CWD.
    """
    source = str(path_or_hf_repo)
    local_path = Path(source).expanduser()
    if local_path.exists():
        if quant:
            raise ValueError("quant is selected by an explicit local artifact")
        return validate_model_path(local_path)
    if isinstance(path_or_hf_repo, Path) or source.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"model directory does not exist: {local_path}")
    spec = model_spec(source, quant)
    repository = spec.repository if spec else source
    subdirectory = spec.subdirectory if spec else ""
    if repository.count("/") != 1:
        raise ValueError(
            f"unknown model {source!r}; provide a catalog name, local directory or owner/repository"
        )
    prefix = f"{subdirectory}/" if subdirectory else ""
    root = Path(
        snapshot_download(
            repo_id=repository,
            allow_patterns=[
                f"{prefix}config.json",
                f"{prefix}weights.npz",
                f"{prefix}weights.safetensors",
            ],
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
    )
    return validate_model_path(root / subdirectory)


def model_artifact_identity(path_or_hf_repo):
    """Identify local artifact replacement cheaply, without rehashing weights."""
    path = Path(path_or_hf_repo).expanduser()
    if not path.exists():
        return ("repository", str(path_or_hf_repo))
    files = []
    for file in (path / "config.json", weight_file(path)):
        info = file.stat()
        files.append((file.name, info.st_size, info.st_mtime_ns, info.st_ino))
    return ("local", str(path.resolve()), tuple(files))


def _apply_alignment_heads(model, heads):
    if isinstance(heads, str):
        heads = heads.encode("ascii")
    elif isinstance(heads, list):
        pairs = np.asarray(heads)
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
        ):
            raise ValueError("alignment_heads must contain integer [layer, head] pairs")
        if (
            not len(pairs)
            or np.any(pairs < 0)
            or np.any(pairs[:, 0] >= model.dims.n_text_layer)
            or np.any(pairs[:, 1] >= model.dims.n_text_head)
        ):
            raise ValueError("alignment_heads contains an out-of-range layer or head")
        heads = pairs
    model.set_alignment_heads(heads)


def pack_attention_weights(weights):
    """Pack source Whisper Q/K/V rows into the canonical inference projections.

    The transformation is in memory only. Quantized packed integers, scales and
    affine offsets concatenate on their output-row axis without requantization.
    Mixing quantized and floating projections inside one group is rejected.
    Existing canonical checkpoints pass through unchanged.
    """
    weights = dict(weights)
    groups = []
    for name in list(weights):
        if name.endswith(".attn.query.weight"):
            groups.append(
                (name.removesuffix(".query.weight"), "qkv", ("query", "key", "value"))
            )
        elif name.endswith(".cross_attn.key.weight"):
            groups.append((name.removesuffix(".key.weight"), "kv", ("key", "value")))
    for prefix, packed_name, sources in groups:
        destination = f"{prefix}.{packed_name}"
        if f"{destination}.weight" in weights:
            raise ValueError(
                f"checkpoint contains both packed and separate attention projections: {prefix}"
            )
        source_names = [f"{prefix}.{source}" for source in sources]
        if any(f"{name}.weight" not in weights for name in source_names):
            raise ValueError(f"incomplete attention projection group: {prefix}")
        source_weights = [weights[f"{name}.weight"] for name in source_names]
        if any(
            value.ndim != 2
            or value.shape != source_weights[0].shape
            or value.dtype != source_weights[0].dtype
            for value in source_weights
        ):
            raise ValueError(
                f"incompatible attention projection shapes/dtypes: {prefix}"
            )
        quantized = [f"{name}.scales" in weights for name in source_names]
        if any(quantized) and not all(quantized):
            raise ValueError(
                f"mixed quantized and floating attention projections: {prefix}"
            )
        if all(quantized) and mx.issubdtype(source_weights[0].dtype, mx.floating):
            raise ValueError(
                f"quantized attention projections require packed integer weights: {prefix}"
            )
        if not any(quantized) and not mx.issubdtype(
            source_weights[0].dtype, mx.floating
        ):
            raise ValueError(
                f"packed integer attention weights require quantization scales: {prefix}"
            )
        weights[f"{destination}.weight"] = mx.concatenate(source_weights, axis=0)
        for suffix in ("scales", "biases"):
            present = [f"{name}.{suffix}" in weights for name in source_names]
            if any(present) and not all(present):
                raise ValueError(f"inconsistent quantization {suffix}: {prefix}")
            if all(present):
                values = [weights[f"{name}.{suffix}"] for name in source_names]
                if any(
                    value.shape[0] != source_weights[0].shape[0]
                    or value.shape[1:] != values[0].shape[1:]
                    or value.dtype != values[0].dtype
                    for value in values
                ):
                    raise ValueError(
                        f"incompatible attention quantization {suffix}: {prefix}"
                    )
                weights[f"{destination}.{suffix}"] = mx.concatenate(values, axis=0)
        ordinary_biases = [weights.get(f"{name}.bias") for name in source_names]
        bias_dtype = next(
            (bias.dtype for bias in ordinary_biases if bias is not None), mx.float32
        )
        biases = []
        for weight, bias in zip(source_weights, ordinary_biases):
            if bias is None:
                bias = mx.zeros((weight.shape[0],), dtype=bias_dtype)
            elif bias.shape != (weight.shape[0],):
                raise ValueError(f"incompatible attention bias shape: {prefix}")
            biases.append(bias)
        weights[f"{destination}.bias"] = mx.concatenate(biases, axis=0)
        for name in source_names:
            for suffix in ("weight", "bias", "scales", "biases"):
                weights.pop(f"{name}.{suffix}", None)
    return weights


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
    *,
    quant=None,
    local_files_only=False,
    cache_dir=None,
    model_name=None,
) -> whisper.Whisper:
    source = model_name or path_or_hf_repo
    spec = None if Path(source).expanduser().exists() else model_spec(source, quant)
    model_path = resolve_model_path(
        path_or_hf_repo,
        quant=quant,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    config = json.loads((model_path / "config.json").read_text())
    config.pop("model_type", None)
    quantization = config.pop("quantization", None)
    alignment_heads = config.pop("alignment_heads", None)
    source_name = config.pop("model_name", None)
    supports_translation = config.pop("supports_translation", None)
    if supports_translation is not None and not isinstance(supports_translation, bool):
        raise ValueError("supports_translation must be a boolean")
    if spec is None and source_name:
        spec = model_spec(source_name)
    model_args = whisper.ModelDimensions(**config)
    weights_path = weight_file(model_path)
    weights = mx.load(str(weights_path))
    # Preserve packed quantized integers while converting floating parameters.
    weights = {
        name: value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
        for name, value in weights.items()
    }
    weights = pack_attention_weights(weights)
    model = whisper.Whisper(model_args, dtype)
    if quantization is not None:
        nn.quantize(
            model,
            **quantization,
            class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights,
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    heads = (
        alignment_heads
        if alignment_heads is not None
        else (spec.alignment_heads if spec else None)
    )
    if heads is not None:
        _apply_alignment_heads(model, heads)
    model.supports_translation = (
        model.is_multilingual
        and (spec is None or spec.translate)
        and supports_translation is not False
    )
    model._model_path = str(model_path)
    digest = hashlib.sha256()
    for filename in (model_path / "config.json", weights_path):
        with filename.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    model._checkpoint_fingerprint = digest.hexdigest()
    return model
