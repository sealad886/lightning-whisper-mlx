from types import SimpleNamespace
from dataclasses import asdict
import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_map

from lightning_whisper_mlx import audio, lightning, transcribe
from lightning_whisper_mlx.load_models import load_model, model_artifact_identity
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("quantized", [False, True])
def test_loaded_checkpoint_honors_precision_without_casting_packed_weights(
    tmp_path, dtype, quantized
):
    dims = ModelDimensions(4, 3, 32, 2, 1, 32, 16, 32, 2, 1)
    model = Whisper(dims, mx.float32)
    saved_dtype = mx.float32 if dtype == mx.float16 else mx.float16
    model.update(tree_map(lambda value: value.astype(saved_dtype), model.parameters()))
    config = asdict(dims)
    if quantized:
        model.decoder.token_embedding = model.decoder.token_embedding.to_quantized(
            group_size=32, bits=4
        )
        config["quantization"] = {"group_size": 32, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.savez(str(tmp_path / "weights.npz"), **dict(tree_flatten(model.parameters())))

    loaded = load_model(str(tmp_path), dtype=dtype)
    assert all(
        value.dtype == dtype
        for _, value in tree_flatten(loaded.parameters())
        if mx.issubdtype(value.dtype, mx.floating)
    )
    if quantized:
        assert isinstance(loaded.decoder.token_embedding, nn.QuantizedEmbedding)
        assert loaded.decoder.token_embedding.weight.dtype == mx.uint32
    features = loaded.encoder(mx.zeros((1, 6, 4), dtype=dtype))
    assert features.dtype == dtype
    assert loaded.logits(mx.array([[1, 2]]), features).shape == (1, 2, 32)


def test_quantization_variants_resolve_distinct_artifacts_and_own_models(
    tmp_path, monkeypatch
):
    from lightning_whisper_mlx.model_catalog import model_spec

    paths = {}
    for quant in (None, "4bit", "8bit"):
        path = tmp_path / (quant or "base")
        path.mkdir()
        (path / "config.json").write_text("{}")
        (path / "weights.npz").write_bytes(b"weights")
        paths[quant] = path
    resolutions = []

    def resolve(source, **options):
        resolutions.append(model_spec(source, options["quant"]).repository)
        return paths[options["quant"]]

    monkeypatch.setattr(lightning, "resolve_model_path", resolve)
    monkeypatch.setattr(
        lightning,
        "load_model",
        lambda *args, **kw: SimpleNamespace(configure_execution=lambda **options: None),
    )
    outputs = []
    monkeypatch.setattr(
        lightning, "transcribe_audio", lambda *a, **kw: outputs.append(kw)
    )
    engines = [
        lightning.LightningWhisperMLX("tiny", quant=q) for q in (None, "4bit", "8bit")
    ]
    assert not resolutions
    for engine in engines:
        engine.transcribe("unused.wav")
    assert len(set(resolutions)) == 3
    assert len({id(output["model_instance"]) for output in outputs}) == 3
    assert [output["path_or_hf_repo"] for output in outputs] == [
        str(path) for path in paths.values()
    ]


@pytest.mark.parametrize("dtypes", [(mx.float16, mx.float32), (mx.float32, mx.float16)])
def test_cache_honors_requested_precision(monkeypatch, dtypes):
    monkeypatch.setattr(transcribe.ModelHolder, "model", None)
    calls = []

    def load(path, dtype):
        calls.append(dtype)
        return SimpleNamespace(dtype=dtype)

    monkeypatch.setattr(transcribe, "load_model", load)
    first = transcribe.ModelHolder.get_model("unavailable-local-fixture", dtypes[0])
    reused = transcribe.ModelHolder.get_model("unavailable-local-fixture", dtypes[0])
    changed = transcribe.ModelHolder.get_model("unavailable-local-fixture", dtypes[1])
    assert reused is first
    assert changed.dtype == dtypes[1]
    assert calls == list(dtypes)


def test_cache_reloads_replaced_local_artifact(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"original")
    monkeypatch.setattr(transcribe.ModelHolder, "model", None)
    loads = []

    def load(*args, **kwargs):
        model = object()
        loads.append(model)
        return model

    monkeypatch.setattr(transcribe, "load_model", load)
    first = transcribe.ModelHolder.get_model(str(tmp_path), mx.float16)
    weights.write_bytes(b"replacement artifact")
    second = transcribe.ModelHolder.get_model(str(tmp_path), mx.float16)
    assert first is not second
    assert len(loads) == 2


def test_artifact_identity_normalizes_equivalent_local_paths(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.npz").write_bytes(b"weights")
    assert model_artifact_identity(tmp_path) == model_artifact_identity(tmp_path / ".")


def test_audio_failure_restores_default_device(monkeypatch):
    original = mx.default_device()
    mx.set_default_device(mx.gpu)

    def fail(*args):
        raise RuntimeError("bad input")

    monkeypatch.setattr(audio, "load_audio", fail)
    try:
        with pytest.raises(RuntimeError, match="bad input"):
            audio.log_mel_spectrogram("invalid.wav")
        assert mx.default_device() == mx.gpu
    finally:
        mx.set_default_device(original)


def test_spectral_failure_restores_default_device(monkeypatch):
    original = mx.default_device()
    mx.set_default_device(mx.gpu)

    def fail(*args, **kwargs):
        raise ValueError("invalid transform")

    monkeypatch.setattr(audio, "stft", fail)
    try:
        with pytest.raises(ValueError, match="invalid transform"):
            audio.log_mel_spectrogram(mx.zeros((800,)))
        assert mx.default_device() == mx.gpu
    finally:
        mx.set_default_device(original)
