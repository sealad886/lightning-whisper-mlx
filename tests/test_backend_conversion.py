import base64
import json
import struct

import numpy as np
import pytest

from benchmarks.convert_mlx_to_ggml import DIMENSIONS, convert, required_tensor_shapes


def test_conversion_maps_convolution_and_records_exact_source(tmp_path):
    model, assets = tmp_path / "model", tmp_path / "assets"
    model.mkdir()
    assets.mkdir()
    dims = {key: 4 for key in DIMENSIONS}
    dims["n_vocab"] = 51864
    (model / "config.json").write_text(json.dumps(dims))
    conv = np.arange(48, dtype=np.float16).reshape(4, 3, 4)
    weights = {
        name.replace(".mlp.0.", ".mlp1.").replace(".mlp.2.", ".mlp2."): np.zeros(
            shape, dtype=np.float16
        )
        for name, shape in required_tensor_shapes(dims).items()
        if name != "encoder.positional_embedding"
    }
    for name in ("encoder.conv1.weight", "encoder.conv2.weight"):
        weights[name] = weights[name].transpose(0, 2, 1)
    weights["encoder.conv1.weight"] = conv
    for name in ("encoder.conv1.bias", "encoder.conv2.bias"):
        weights[name] = weights[name].reshape(-1)
    weights = {"encoder.conv1.weight": weights.pop("encoder.conv1.weight"), **weights}
    np.savez(model / "weights.npz", **weights)
    np.savez(assets / "mel_filters.npz", mel_4=np.ones((4, 3), dtype=np.float32))
    (assets / "gpt2.tiktoken").write_bytes(base64.b64encode(b"a") + b" 0\n")
    output = tmp_path / "converted.bin"
    receipt = convert(model, output, assets=assets)
    assert receipt["source_files"]["weights.npz"]
    raw = output.read_bytes()
    # 48-byte header; 8-byte filter shape + 24 bytes; 4 token count + 4 length + 1 token.
    offset = 48 + 8 + 48 + 4 + 4 + 1
    ndim, nlen, ftype = struct.unpack_from("<3i", raw, offset)
    assert (ndim, ftype) == (3, 1)
    offset += 12
    assert struct.unpack_from("<3i", raw, offset) == (3, 4, 4)
    offset += 12
    assert raw[offset : offset + nlen] == b"encoder.conv1.weight"
    data = np.frombuffer(
        raw[offset + nlen : offset + nlen + conv.nbytes], dtype="<f2"
    ).reshape(4, 4, 3)
    np.testing.assert_array_equal(data, conv.transpose(0, 2, 1))
    with pytest.raises(FileExistsError):
        convert(model, output, assets=assets)


def test_conversion_rejects_quantized_checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    with pytest.raises(ValueError, match="unquantized"):
        convert(model, tmp_path / "out.bin")


def test_packed_and_separate_checkpoints_produce_identical_ggml_bytes(tmp_path):
    # Observable conversion contract: equivalent models have identical GGML bytes,
    # including Q/V biases and both self- and cross-attention projections.
    dims = {key: 4 for key in DIMENSIONS}
    dims.update(n_vocab=64, n_audio_layer=1, n_text_layer=1)
    assets = tmp_path / "assets"
    assets.mkdir()
    np.savez(assets / "mel_filters.npz", mel_4=np.ones((4, 3), dtype=np.float32))
    (assets / "gpt2.tiktoken").write_bytes(b"YQ== 0\n")
    rng = np.random.default_rng(7)
    separate = {
        name.replace(".mlp.0.", ".mlp1.")
        .replace(".mlp.2.", ".mlp2."): rng.normal(size=shape)
        .astype(np.float16)
        for name, shape in required_tensor_shapes(dims).items()
        if name != "encoder.positional_embedding"
    }
    for name in ("encoder.conv1.weight", "encoder.conv2.weight"):
        separate[name] = separate[name].transpose(0, 2, 1)
    for name in ("encoder.conv1.bias", "encoder.conv2.bias"):
        separate[name] = separate[name].reshape(-1)
    packed = dict(separate)
    for prefix, storage, projections in (
        ("encoder.blocks.0.attn", "qkv", ("query", "key", "value")),
        ("decoder.blocks.0.attn", "qkv", ("query", "key", "value")),
        ("decoder.blocks.0.cross_attn", "kv", ("key", "value")),
    ):
        packed[f"{prefix}.{storage}.weight"] = np.concatenate(
            [packed.pop(f"{prefix}.{p}.weight") for p in projections]
        )
        packed[f"{prefix}.{storage}.bias"] = np.concatenate(
            [
                (
                    np.zeros(4, dtype=np.float16)
                    if p == "key"
                    else packed.pop(f"{prefix}.{p}.bias")
                )
                for p in projections
            ]
        )
    for label, weights in (("separate", separate), ("packed", packed)):
        model = tmp_path / label
        model.mkdir()
        (model / "config.json").write_text(json.dumps(dims))
        np.savez(model / "weights.npz", **weights)
        convert(model, tmp_path / f"{label}.bin", assets=assets)
    assert (tmp_path / "separate.bin").read_bytes() == (
        tmp_path / "packed.bin"
    ).read_bytes()
    # A nonzero packed key bias changes the model and must not be silently dropped.
    packed["decoder.blocks.0.cross_attn.kv.bias"][0] = 1
    np.savez(tmp_path / "packed/weights.npz", **packed)
    with pytest.raises(ValueError, match="key bias must be exactly zero"):
        convert(tmp_path / "packed", tmp_path / "bad.bin", assets=assets)
    assert not (tmp_path / "bad.bin").exists()
