"""Convert an unquantized local MLX Whisper checkpoint into whisper.cpp GGML.

Writes a new artifact and provenance receipt; never overwrites source weights.
Format follows whisper.cpp/models/convert-pt-to-ggml.py, with MLX convolution and
MLP names mapped back to Whisper tensor conventions. Packed QKV/KV rows are
expanded after validating the key bias is exactly zero. No Torch/MLX import needed.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import struct
import tempfile
import os
import math

from .cpp_backend import file_digest

DIMENSIONS = (
    "n_vocab",
    "n_audio_ctx",
    "n_audio_state",
    "n_audio_head",
    "n_audio_layer",
    "n_text_ctx",
    "n_text_state",
    "n_text_head",
    "n_text_layer",
    "n_mels",
)


def required_tensor_shapes(config):
    shapes = {}
    for kind in ("encoder", "decoder"):
        width, layers = (
            config[f"n_{'audio' if kind == 'encoder' else 'text'}_state"],
            config[f"n_{'audio' if kind == 'encoder' else 'text'}_layer"],
        )
        for index in range(layers):
            prefix = f"{kind}.blocks.{index}."
            for attention in (
                ("attn",) if kind == "encoder" else ("attn", "cross_attn")
            ):
                for projection in ("query", "key", "value", "out"):
                    shapes[prefix + attention + "." + projection + ".weight"] = (
                        width,
                        width,
                    )
                    if projection != "key":
                        shapes[prefix + attention + "." + projection + ".bias"] = (
                            width,
                        )
                for parameter in ("weight", "bias"):
                    shapes[prefix + attention + "_ln." + parameter] = (width,)
            shapes[prefix + "mlp.0.weight"] = (4 * width, width)
            shapes[prefix + "mlp.0.bias"] = (4 * width,)
            shapes[prefix + "mlp.2.weight"] = (width, 4 * width)
            shapes[prefix + "mlp.2.bias"] = (width,)
            for parameter in ("weight", "bias"):
                shapes[prefix + "mlp_ln." + parameter] = (width,)
    audio, text = config["n_audio_state"], config["n_text_state"]
    shapes.update(
        {
            "encoder.conv1.weight": (audio, config["n_mels"], 3),
            "encoder.conv2.weight": (audio, audio, 3),
            "encoder.conv1.bias": (audio, 1),
            "encoder.conv2.bias": (audio, 1),
            "encoder.positional_embedding": (config["n_audio_ctx"], audio),
            "decoder.positional_embedding": (config["n_text_ctx"], text),
            "decoder.token_embedding.weight": (config["n_vocab"], text),
        }
    )
    for kind, layer, width in (("encoder", "ln_post", audio), ("decoder", "ln", text)):
        for parameter in ("weight", "bias"):
            shapes[f"{kind}.{layer}.{parameter}"] = (width,)
    return shapes


def unpack_attention(weights, config):
    """Expand canonical packed MLX rows once for the external GGML artifact."""
    import numpy as np

    weights = dict(weights)
    for name in tuple(weights):
        if name.endswith(".attn.qkv.weight"):
            prefix, packed, projections = (
                name.removesuffix(".qkv.weight"),
                "qkv",
                ("query", "key", "value"),
            )
        elif name.endswith(".cross_attn.kv.weight"):
            prefix, packed, projections = (
                name.removesuffix(".kv.weight"),
                "kv",
                ("key", "value"),
            )
        else:
            continue
        width = config[
            "n_audio_state" if prefix.startswith("encoder.") else "n_text_state"
        ]
        bias_name = f"{prefix}.{packed}.bias"
        weight, bias = weights[name], weights.get(bias_name)
        if (
            weight.dtype.kind != "f"
            or weight.shape != (len(projections) * width, width)
            or bias is None
            or bias.dtype.kind != "f"
            or bias.shape != (len(projections) * width,)
        ):
            raise ValueError(f"invalid packed attention shapes/dtypes: {prefix}")
        if any(
            f"{prefix}.{projection}.{parameter}" in weights
            for projection in projections
            for parameter in ("weight", "bias")
        ):
            raise ValueError(f"mixed packed and separate attention tensors: {prefix}")
        for index, projection in enumerate(projections):
            begin, end = index * width, (index + 1) * width
            weights[f"{prefix}.{projection}.weight"] = weight[begin:end]
            if projection == "key":
                if np.count_nonzero(bias[begin:end]):
                    raise ValueError(f"packed key bias must be exactly zero: {prefix}")
            else:
                weights[f"{prefix}.{projection}.bias"] = bias[begin:end]
        del weights[name], weights[bias_name]
    return weights


def tensor_order(item):
    """Stable GGML bytes independent of NPZ insertion order and packed storage."""
    name = item[0]
    base, _, parameter = name.rpartition(".")
    return (
        0 if name.startswith("encoder.conv") else 1,
        base,
        {"weight": 0, "bias": 1}.get(parameter, 2),
        parameter,
    )


def convert(model, output, *, assets=None):
    import numpy as np

    model, output = Path(model).resolve(), Path(output).resolve()
    assets = Path(
        assets or Path(__file__).resolve().parents[1] / "lightning_whisper_mlx/assets"
    )
    receipt_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("output model or receipt already exists")
    config = json.loads((model / "config.json").read_text())
    if config.get("quantization") is not None:
        raise ValueError("only unquantized checkpoints can be converted losslessly")
    if any(
        not isinstance(config.get(key), int) or config[key] <= 0 for key in DIMENSIONS
    ):
        raise ValueError("invalid model dimensions")
    token_path = assets / (
        "multilingual.tiktoken" if config["n_vocab"] >= 51865 else "gpt2.tiktoken"
    )
    tokens = sorted(
        (int(rank), (b"" if token == b"=" else base64.b64decode(token, validate=True)))
        for token, rank in (
            line.split() for line in token_path.read_bytes().splitlines() if line
        )
    )
    if [rank for rank, _ in tokens] != list(range(len(tokens))):
        raise ValueError("tokenizer ranks must be contiguous")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream, np.load(
            model / "weights.npz", allow_pickle=False
        ) as weights, np.load(
            assets / "mel_filters.npz", allow_pickle=False
        ) as filters:
            stream.write(
                struct.pack("<12i", 0x67676D6C, *(config[key] for key in DIMENSIONS), 1)
            )
            mel = np.asarray(filters[f"mel_{config['n_mels']}"], dtype="<f4")
            stream.write(struct.pack("<2i", *mel.shape))
            stream.write(mel.tobytes())
            stream.write(struct.pack("<i", len(tokens)))
            for _, token in tokens:
                stream.write(struct.pack("<i", len(token)))
                stream.write(token)
            mapped = set()
            expected = required_tensor_shapes(config)
            width = config["n_audio_state"]
            if width < 4 or width % 2:
                raise ValueError("audio state must be even and >=4")
            # MLX deliberately excludes fixed sinusoids from saved parameters.
            # Reconstruct the same float32 formula as Whisper's reference model.
            inverse = np.exp(
                -np.float32(math.log(10000) / (width // 2 - 1))
                * np.arange(width // 2, dtype=np.float32)
            )
            time = (
                np.arange(config["n_audio_ctx"], dtype=np.float32)[:, None]
                * inverse[None, :]
            )
            positions = np.concatenate((np.sin(time), np.cos(time)), axis=1)
            tensors = list(
                unpack_attention(
                    {key: weights[key] for key in weights.files}, config
                ).items()
            )
            if "encoder.positional_embedding" not in weights.files:
                tensors.append(("encoder.positional_embedding", positions))
            for original, tensor in sorted(tensors, key=tensor_order):
                if original == "alignment_heads":
                    if (
                        tensor.ndim != 2
                        or tensor.shape[1] != 2
                        or tensor.dtype.kind not in "iu"
                    ):
                        raise ValueError("invalid alignment-head metadata")
                    continue  # Not a model weight; word alignment is a separate workload.
                if tensor.dtype.kind != "f":
                    raise ValueError(f"non-floating tensor is unsupported: {original}")
                name_key = original.replace(".mlp1.", ".mlp.0.").replace(
                    ".mlp2.", ".mlp.2."
                )
                if name_key in mapped:
                    raise ValueError("duplicate converted tensor")
                mapped.add(name_key)
                if name_key in {"encoder.conv1.weight", "encoder.conv2.weight"}:
                    if tensor.ndim != 3:
                        raise ValueError(
                            "convolution weight must have three dimensions"
                        )
                    tensor = tensor.transpose(0, 2, 1)
                tensor = tensor.squeeze()
                if name_key in {"encoder.conv1.bias", "encoder.conv2.bias"}:
                    tensor = tensor.reshape(-1, 1)
                if name_key not in expected or tensor.shape != expected[name_key]:
                    raise ValueError(
                        f"unexpected tensor or shape: {name_key} {tensor.shape}"
                    )
                fp32 = tensor.ndim < 2 or name_key in {
                    "encoder.conv1.bias",
                    "encoder.conv2.bias",
                    "encoder.positional_embedding",
                    "decoder.positional_embedding",
                }
                tensor = np.ascontiguousarray(tensor, dtype="<f4" if fp32 else "<f2")
                encoded = name_key.encode()
                stream.write(
                    struct.pack("<3i", tensor.ndim, len(encoded), 0 if fp32 else 1)
                )
                stream.write(
                    struct.pack("<" + "i" * tensor.ndim, *reversed(tensor.shape))
                )
                stream.write(encoded)
                stream.write(tensor.tobytes())
            if mapped != expected.keys():
                raise ValueError(
                    "missing model tensors: "
                    + ", ".join(sorted(expected.keys() - mapped))
                )
        # Exclusive creation prevents races from overwriting an existing artifact.
        os.link(name, output)
        receipt = {
            "format": "whisper.cpp-ggml-f16",
            "source_model": str(model),
            "source_files": {
                key: file_digest(model / key) for key in ("config.json", "weights.npz")
            },
            "assets": {
                key: file_digest(path)
                for key, path in {
                    "tokenizer": token_path,
                    "mel_filters": assets / "mel_filters.npz",
                }.items()
            },
            "output_sha256": file_digest(output),
            "dimensions": {key: config[key] for key in DIMENSIONS},
            "encoder_positions": "fixed float32 reference sinusoids reconstructed because MLX excludes them from NPZ",
            "conversion": "MLX conv[out,k,in] -> Whisper[out,in,k]; packed QKV/KV split; zero key bias verified/omitted; MLP names; upstream GGML f16/f32 tensor policy; canonical tensor order",
        }
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        return receipt
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args(argv)
    convert(args.model, args.output, assets=args.assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
