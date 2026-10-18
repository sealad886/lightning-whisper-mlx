"""Direct comparison contract, with CPU-only stub engines."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks import compare_backends


@pytest.mark.parametrize("require_uncontended", [False, True])
@pytest.mark.parametrize("joint_chunk", [None, 1, 4])
def test_cli_resident_engines_warmups_refs_and_round_robin(
    monkeypatch, tmp_path, require_uncontended, joint_chunk
):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fixture")
    references = tmp_path / "refs.json"
    references.write_text('["hello"]')
    output = tmp_path / "report.json"
    config = {"n_vocab": 51865, "n_audio_layer": 6, "n_text_layer": 6}
    monkeypatch.setattr(
        compare_backends,
        "validate_artifacts",
        lambda args: (config, {"common_checkpoint_verified": True}),
    )
    monkeypatch.setattr(
        compare_backends, "decode_pcm", lambda path: np.zeros(16000, dtype=np.float32)
    )
    monkeypatch.setattr(
        compare_backends, "model_suppression_ids", lambda *args: (1, 2, 3)
    )

    class Guard:
        def __init__(self, policy, required=False):
            self.required = required
            self.owned_pids = set()

        def check(self, *args, **kwargs):
            return {"status": "observed-low-contention"}

        def summary(self):
            return {
                "status": "observed-low-contention",
                "qualified": self.required,
                "required": self.required,
                "guarantees_uncontended": False,
            }

    monkeypatch.setattr(compare_backends, "QuietGuard", Guard)
    calls = []
    configured = []

    class Model:
        def parameters(self):
            return []

        def configure_execution(self, **kwargs):
            configured.append(kwargs)

    model = Model()

    class Backend:
        ready = {"dimensions": config}

        def __init__(self, *args, **kwargs):
            assert kwargs["suppress_tokens"] == (1, 2, 3)
            calls.append("cpp-init")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("cpp-close")

        def upload(self, *args):
            calls.append("upload")

        def transcribe(self, clip):
            calls.append("cpp")
            return {
                "text": "hello",
                "language": "en",
                "segments": [],
                "tokens": [],
                "service_seconds": 0.001,
            }

    monkeypatch.setattr(compare_backends, "CppBackend", Backend)
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.float16 = "float16"
    core.eval = lambda value: None
    core.synchronize = lambda: None
    mlx.core = core
    loader = ModuleType("lightning_whisper_mlx.load_models")

    def load(*args, **kwargs):
        calls.append("mlx-init")
        return model

    loader.load_model = load
    transcribe = ModuleType("lightning_whisper_mlx.transcribe")

    def run(*args, **kwargs):
        assert kwargs["model_instance"] is model
        assert kwargs["temperature"] == 0
        assert kwargs["compression_ratio_threshold"] is None
        assert kwargs["batch_size"] == 1
        assert kwargs["suppress_tokens"] == [1, 2, 3]
        calls.append("mlx")
        return {"text": "hello", "language": "en", "segments": []}

    transcribe.transcribe_audio = run
    for name, module in {
        "mlx": mlx,
        "mlx.core": core,
        "lightning_whisper_mlx.load_models": loader,
        "lightning_whisper_mlx.transcribe": transcribe,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    assert (
        compare_backends.main(
            [
                *(["--require-uncontended"] if require_uncontended else []),
                *(
                    [
                        "--compile-decoder",
                        "--joint-decoding",
                        "--decode-chunk-size",
                        str(joint_chunk),
                    ]
                    if joint_chunk
                    else []
                ),
                "--mlx-model",
                str(tmp_path),
                "--cpp-model",
                "unused",
                "--worker",
                "unused",
                "--checkpoint",
                "base",
                "--audio",
                str(audio),
                "--references",
                str(references),
                "--language",
                "en",
                "--without-timestamps",
                "--warmups",
                "1",
                "--repetitions",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["assessment"]["speed_claim_qualified"] == require_uncontended
    assert calls.count("mlx-init") == calls.count("cpp-init") == 1
    assert calls.count("upload") == 1
    assert calls.count("mlx") == calls.count("cpp") == 3
    assert calls[-1] == "cpp-close"
    if joint_chunk:
        assert configured[0]["joint_decoding"]
        assert configured[0]["decode_chunk_size"] == joint_chunk
    options = report["results"][0]["metadata"]["options"]
    assert options["mlx_joint_decoding"] == bool(joint_chunk)
    assert options["mlx_decode_chunk_size"] == (joint_chunk or 1)
    assert options["mlx_constant_pcm_rejection"]
    assert report["config"]["execution_policy"] == "interleaved-rotating-start"


def test_output_existing_rejected_before_loading(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("untouched")
    with pytest.raises(SystemExit):
        compare_backends.main(
            [
                "--mlx-model",
                "unused",
                "--cpp-model",
                "unused",
                "--worker",
                "unused",
                "--checkpoint",
                "base",
                "--audio",
                "unused",
                "--language",
                "en",
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "untouched"


def test_canonical_suppression_includes_default_non_speech_and_special_tokens():
    tokenizer = SimpleNamespace(
        non_speech_tokens=(3268, 4, 4),
        transcribe=10,
        translate=11,
        sot=12,
        sot_prev=13,
        sot_lm=14,
        no_speech=15,
    )
    assert compare_backends.canonical_suppression_ids(tokenizer, 5000) == (
        4,
        10,
        11,
        12,
        13,
        14,
        15,
        3268,
    )
    tokenizer.no_speech = None
    assert 15 not in compare_backends.canonical_suppression_ids(tokenizer, 5000)
    with pytest.raises(ValueError, match="vocabulary"):
        compare_backends.canonical_suppression_ids(tokenizer, 10)


def test_conversion_asset_receipt_must_match_runtime(tmp_path):
    from benchmarks.cpp_backend import file_digest

    (tmp_path / "multilingual.tiktoken").write_bytes(b"runtime-vocabulary")
    (tmp_path / "mel_filters.npz").write_bytes(b"runtime-filters")
    assets = {
        "tokenizer": file_digest(tmp_path / "multilingual.tiktoken"),
        "mel_filters": file_digest(tmp_path / "mel_filters.npz"),
    }
    assert (
        compare_backends.verify_conversion_assets(
            {"assets": assets}, {"n_vocab": 51865}, assets=tmp_path
        )
        == assets
    )
    for changed in ("tokenizer", "mel_filters"):
        with pytest.raises(ValueError, match="runtime assets"):
            compare_backends.verify_conversion_assets(
                {"assets": {**assets, changed: "other"}},
                {"n_vocab": 51865},
                assets=tmp_path,
            )


def test_common_model_policy_adds_all_languages_and_checks_canonical_decoder(
    monkeypatch,
):
    tokenizer = SimpleNamespace(
        non_speech_tokens=(3268, 4),
        transcribe=10,
        translate=11,
        sot=12,
        sot_prev=13,
        sot_lm=14,
        no_speech=15,
        all_language_tokens=(20, 21),
    )
    token_module = ModuleType("lightning_whisper_mlx.tokenizer")
    token_module.get_tokenizer = lambda *args, **kwargs: tokenizer
    decode_module = ModuleType("lightning_whisper_mlx.decoding")
    decode_module.DecodingTask = SimpleNamespace(
        _get_suppress_tokens=lambda task: (4, 10, 11, 12, 13, 14, 15, 3268)
    )
    monkeypatch.setitem(sys.modules, "lightning_whisper_mlx.tokenizer", token_module)
    monkeypatch.setitem(sys.modules, "lightning_whisper_mlx.decoding", decode_module)
    model = SimpleNamespace(
        is_multilingual=True, num_languages=99, dims=SimpleNamespace(n_vocab=5000)
    )
    assert compare_backends.model_suppression_ids(model, "en", "transcribe") == (
        4,
        10,
        11,
        12,
        13,
        14,
        15,
        20,
        21,
        3268,
    )
