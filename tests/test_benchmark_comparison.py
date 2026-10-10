import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from benchmarks.compare_decoding import assess_results, main


def result(mode, outputs, seconds):
    return SimpleNamespace(
        case=SimpleNamespace(mode=mode, lookahead=2),
        outputs=outputs,
        median_seconds=seconds,
    )


def clip(**changes):
    return {
        "tokens": [1, 2],
        "text": "word",
        "language": "en",
        "avg_logprob": -0.25,
        "no_speech_prob": 0.05,
        **changes,
    }


def test_comparison_checks_all_repetitions_including_baseline():
    good = [clip()]
    bad = [clip(tokens=[1, 3])]
    baseline = result("none", [good, good], 2)
    candidate = result("assistant", [bad, good], 1)
    assessment = assess_results([baseline, candidate])
    assert not assessment["comparisons"][0]["exact_match"]
    assert assessment["comparisons"][0]["median_speedup"] == 2
    baseline.outputs = [bad, good]
    assert not assess_results([baseline, candidate])["baseline_stable"]


def test_exactness_and_speed_are_separate_gates():
    value = [clip()]
    assessment = assess_results(
        [
            result("none", [value], 1),
            result("assistant", [value], 2),
        ]
    )
    assert assessment["comparisons"][0]["exact_match"]
    assert not assessment["comparisons"][0]["meets_speed_threshold"]


def test_float_reduction_noise_does_not_break_exact_token_comparison():
    value = [clip()]
    perturbed = [clip(avg_logprob=-0.250001, no_speech_prob=0.050001)]
    assessment = assess_results(
        [
            result("none", [value, perturbed], 2),
            result("assistant", [perturbed, value], 1),
        ]
    )

    assert assessment["baseline_stable"]
    assert assessment["baseline_scores_stable"]
    assert assessment["comparisons"][0]["exact_match"]
    assert assessment["comparisons"][0]["decision_scores_match"]
    assert assessment["score_tolerance"]["absolute"] > 0


@pytest.mark.parametrize("field", ["avg_logprob", "no_speech_prob"])
def test_changed_decision_score_fails_even_with_exact_fast_tokens(field):
    value = [clip()]
    changed = [clip(**{field: 0.9})]
    baseline = result("none", [value, value], 2)
    candidate = result("assistant", [changed, value], 1)
    assessment = assess_results([baseline, candidate])

    assert assessment["comparisons"][0]["exact_match"]
    assert assessment["comparisons"][0]["meets_speed_threshold"]
    assert not assessment["comparisons"][0]["decision_scores_match"]
    baseline.outputs = [changed, value]
    assert not assess_results([baseline, candidate])["baseline_scores_stable"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), None])
def test_nonfinite_or_missing_decision_scores_cannot_qualify(score):
    value = [clip()]
    changed = [clip(avg_logprob=score)]
    assessment = assess_results(
        [
            result("none", [value], 2),
            result("mtp", [changed], 1),
        ]
    )

    assert assessment["comparisons"][0]["exact_match"]
    assert not assessment["comparisons"][0]["decision_scores_match"]


@pytest.mark.parametrize("changed_score, expected_exit", [(False, 0), (True, 1)])
def test_comparison_cli_preserves_scores_and_scoped_mode_memory(
    tmp_path, monkeypatch, changed_score, expected_exit
):
    # Exercise the CLI contract with synthetic dependency modules: no MLX import,
    # model inference, or native allocator measurement takes place in this test.
    def module(name, **values):
        value = ModuleType(name)
        value.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    runtime_state = {"active": 100, "peak": 0, "resets": 0}

    def reset_peak():
        runtime_state["peak"] = 0
        runtime_state["resets"] += 1

    runtime = module(
        "mlx.core",
        float16="float16",
        eval=lambda value: None,
        synchronize=lambda: None,
        reset_peak_memory=reset_peak,
        get_active_memory=lambda: runtime_state["active"],
        get_peak_memory=lambda: runtime_state["peak"],
        get_cache_memory=lambda: 20,
    )
    module("mlx", core=runtime)
    module("lightning_whisper_mlx", __path__=[])
    model = SimpleNamespace(
        is_multilingual=True,
        num_languages=99,
        _checkpoint_fingerprint="synthetic",
        dims=SimpleNamespace(n_vocab=100, n_mels=80, n_text_ctx=448),
    )
    module("lightning_whisper_mlx.load_models", load_model=lambda *a, **kw: model)
    module(
        "lightning_whisper_mlx.tokenizer", get_tokenizer=lambda *a, **kw: "tokenizer"
    )
    module(
        "lightning_whisper_mlx.mtp",
        fingerprint_tokenizer=lambda value: "synthetic",
        load_mtp_head=lambda *a, **kw: None,
    )
    module(
        "lightning_whisper_mlx.audio",
        N_SAMPLES=480000,
        SAMPLE_RATE=16000,
        load_audio=lambda path: SimpleNamespace(size=16000),
        pad_or_trim=lambda audio: audio,
        log_mel_spectrogram=lambda audio, **kw: "synthetic mel",
    )

    def decode(model, mel, options):
        runtime_state["peak"] = 150 if options.speculative_mode == "none" else 250
        values = clip()
        if changed_score and options.speculative_mode == "assistant":
            values["avg_logprob"] = -0.9
        return SimpleNamespace(**values)

    module(
        "lightning_whisper_mlx.decoding", DecodingOptions=SimpleNamespace, decode=decode
    )
    for name in ("target", "assistant"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "weights.npz").write_bytes(b"synthetic")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"synthetic")
    output = tmp_path / "out.json"

    assert (
        main(
            [
                "--target",
                str(tmp_path / "target"),
                "--assistant",
                str(tmp_path / "assistant"),
                "--audio",
                str(audio),
                "--language",
                "en",
                "--warmups",
                "1",
                "--repetitions",
                "2",
                "--output",
                str(output),
            ]
        )
        == expected_exit
    )
    payload = json.loads(output.read_text())
    assert runtime_state["resets"] == 4
    assert [record["peak_memory_bytes"] for record in payload["results"]] == [150, 250]
    assert payload["peak_memory_bytes"] == 250
    assert all(
        "all selected artifacts resident" in record["metadata"]["memory_scope"]
        for record in payload["results"]
    )
    assert payload["results"][1]["outputs"][0][0]["avg_logprob"] == (
        -0.9 if changed_score else -0.25
    )
    assert all(
        sample["active_memory_before_bytes"] == 100
        and sample["cached_memory_after_bytes"] == 20
        for record in payload["results"]
        for sample in record["sample_metrics"]
    )


def test_missing_local_assets_fail_before_runtime_loading(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--target",
                str(tmp_path / "target"),
                "--assistant",
                str(tmp_path / "assistant"),
                "--audio",
                str(tmp_path / "speech.wav"),
                "--language",
                "en",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert error.value.code == 2
