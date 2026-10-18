"""Artifact resolution, session lifetime, and model metadata contracts."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from lightning_whisper_mlx import lightning
from lightning_whisper_mlx import load_models
from lightning_whisper_mlx.model_catalog import MODEL_CATALOG, model_spec
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def artifact(path, weights="weights.npz"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / weights).write_bytes(b"weights")
    return path


@pytest.mark.parametrize(
    "name",
    [
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "turbo",
        "large-v3-turbo",
    ],
)
def test_official_names_have_explicit_artifacts_and_alignment(name):
    spec = model_spec(name)
    assert spec.repository.startswith("mlx-community/")
    assert spec.alignment_heads


def test_large_alias_is_current_official_large_and_turbo_cannot_translate():
    assert MODEL_CATALOG["large"] == MODEL_CATALOG["large-v3"]
    assert not model_spec("turbo").translate
    assert model_spec("large-v3").translate


def test_catalog_rejects_unpublished_quantization():
    with pytest.raises(ValueError, match="unavailable"):
        model_spec("turbo", "4bit")


@pytest.mark.parametrize("filename", ["weights.npz", "weights.safetensors"])
def test_local_artifacts_are_used_in_place_without_network(
    tmp_path, monkeypatch, filename
):
    model_path = artifact(tmp_path / "model", filename)
    monkeypatch.setattr(
        load_models, "snapshot_download", lambda **kw: pytest.fail("unexpected network")
    )
    assert (
        load_models.resolve_model_path(model_path, local_files_only=True)
        == model_path.resolve()
    )
    assert load_models.model_artifact_identity(model_path)[2][1][0] == filename


@pytest.mark.parametrize("missing", ["config.json", "weights.npz"])
def test_incomplete_local_artifact_fails_without_remote_fallback(
    tmp_path, monkeypatch, missing
):
    model_path = artifact(tmp_path / "model")
    (model_path / missing).unlink()
    monkeypatch.setattr(
        load_models, "snapshot_download", lambda **kw: pytest.fail("unexpected network")
    )
    with pytest.raises(FileNotFoundError):
        load_models.resolve_model_path(model_path)


def test_missing_explicit_path_does_not_attempt_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        load_models, "snapshot_download", lambda **kw: pytest.fail("unexpected network")
    )
    with pytest.raises(FileNotFoundError):
        load_models.resolve_model_path(tmp_path / "missing")


def test_hub_resolution_respects_offline_and_cache_directory(tmp_path, monkeypatch):
    model_path = artifact(tmp_path / "snapshot")
    calls = []
    monkeypatch.setattr(
        load_models, "snapshot_download", lambda **kw: calls.append(kw) or model_path
    )
    assert (
        load_models.resolve_model_path(
            "turbo", local_files_only=True, cache_dir="/chosen/cache"
        )
        == model_path.resolve()
    )
    assert calls[0]["repo_id"] == "mlx-community/whisper-large-v3-turbo"
    assert calls[0]["local_files_only"] is True
    assert calls[0]["cache_dir"] == "/chosen/cache"
    assert "local_dir" not in calls[0]
    assert "weights.safetensors" in calls[0]["allow_patterns"]


def test_distilled_nested_artifact_resolves_without_copying(tmp_path, monkeypatch):
    nested = artifact(tmp_path / "mlx_models" / "distil-small.en")
    monkeypatch.setattr(load_models, "snapshot_download", lambda **kw: tmp_path)
    assert load_models.resolve_model_path("distil-small.en") == nested.resolve()


def test_session_resident_models_do_not_evict_each_other_and_reload_replaced_artifacts(
    tmp_path, monkeypatch
):
    paths = [artifact(tmp_path / name) for name in ("first", "second")]
    loads = []

    def load(source, **options):
        value = SimpleNamespace(
            dtype=options["dtype"], configure_execution=lambda **kw: None
        )
        loads.append(value)
        return value

    monkeypatch.setattr(lightning, "load_model", load)
    first, second = [
        lightning.LightningWhisperMLX(path, local_files_only=True) for path in paths
    ]
    assert not loads
    one, two = first.model, second.model
    assert first.model is one
    assert second.model is two
    assert len(loads) == 2
    (paths[0] / "weights.npz").write_bytes(b"replacement")
    assert first.model is not one
    assert second.model is two
    assert len(loads) == 3


def test_session_uses_requested_precision_and_passes_its_resident_model(
    tmp_path, monkeypatch
):
    path = artifact(tmp_path / "model")
    loads, calls = [], []
    monkeypatch.setattr(
        lightning,
        "load_model",
        lambda *a, **kw: loads.append(kw["dtype"])
        or SimpleNamespace(configure_execution=lambda **options: None),
    )
    monkeypatch.setattr(
        lightning, "transcribe_audio", lambda *a, **kw: calls.append(kw) or "result"
    )
    engine = lightning.LightningWhisperMLX(path, dtype=mx.float32)
    assert engine.transcribe("audio.wav") == "result"
    resident = calls[-1]["model_instance"]
    engine.transcribe("audio.wav")
    assert calls[-1]["model_instance"] is resident
    assert loads == [mx.float32]
    engine.transcribe("audio.wav", fp16=True)
    assert loads == [mx.float32, mx.float16]


def test_turbo_translation_fails_before_loading_or_downloading(monkeypatch):
    monkeypatch.setattr(
        lightning,
        "resolve_model_path",
        lambda *a, **kw: pytest.fail("unexpected download"),
    )
    engine = lightning.LightningWhisperMLX("turbo")
    with pytest.raises(ValueError, match="translation"):
        engine.transcribe("audio.wav", task="translate")


@pytest.mark.parametrize(
    "heads", [[[0, 0], [1, 1]], "ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00"]
)
def test_loader_reads_explicit_alignment_metadata(tmp_path, heads):
    # tiny.en dimensions only for its published mask; no full model allocation.
    layers, n_heads = (4, 6) if isinstance(heads, str) else (2, 2)
    dims = ModelDimensions(4, 3, 12, 2, 1, 32, 16, 12, n_heads, layers)
    model = Whisper(dims, mx.float32)
    config = {**asdict(dims), "alignment_heads": heads, "supports_translation": False}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.savez(str(tmp_path / "weights.npz"), **dict(tree_flatten(model.parameters())))
    loaded = load_models.load_model(tmp_path)
    assert not loaded.supports_translation
    if isinstance(heads, list):
        assert np.array_equal(np.array(loaded.alignment_heads), np.asarray(heads))
    else:
        assert loaded.alignment_heads.shape[1] == 2
        assert loaded.alignment_heads.shape[0] > 0


def test_safetensors_checkpoint_loads_without_conversion(tmp_path):
    dims = ModelDimensions(4, 3, 8, 2, 1, 32, 16, 8, 2, 1)
    model = Whisper(dims, mx.float32)
    (tmp_path / "config.json").write_text(json.dumps(asdict(dims)))
    mx.save_safetensors(
        str(tmp_path / "weights.safetensors"), dict(tree_flatten(model.parameters()))
    )
    loaded = load_models.load_model(tmp_path, dtype=mx.float16)
    assert loaded.encoder(mx.zeros((1, 6, 4), dtype=mx.float16)).dtype == mx.float16
