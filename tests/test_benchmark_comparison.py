from types import SimpleNamespace

import pytest

from benchmarks.compare_decoding import assess_results, main


def result(mode, outputs, seconds):
    return SimpleNamespace(
        case=SimpleNamespace(mode=mode, lookahead=2),
        outputs=outputs,
        median_seconds=seconds,
    )


def test_comparison_checks_all_repetitions_including_baseline():
    good = [{"tokens": [1, 2], "text": "word"}]
    bad = [{"tokens": [1, 3], "text": "word"}]
    baseline = result("none", [good, good], 2)
    candidate = result("assistant", [bad, good], 1)
    assessment = assess_results([baseline, candidate])
    assert not assessment["comparisons"][0]["exact_match"]
    assert assessment["comparisons"][0]["median_speedup"] == 2
    baseline.outputs = [bad, good]
    assert not assess_results([baseline, candidate])["baseline_stable"]


def test_exactness_and_speed_are_separate_gates():
    value = [{"tokens": [1, 2], "text": "word"}]
    assessment = assess_results(
        [
            result("none", [value], 1),
            result("assistant", [value], 2),
        ]
    )
    assert assessment["comparisons"][0]["exact_match"]
    assert not assessment["comparisons"][0]["meets_speed_threshold"]


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
