from types import SimpleNamespace

import pytest

from benchmarks.quality import (
    assess_backend_results,
    normalize_text,
    text_quality,
    summarize_times,
    edit_distance,
)


def result(texts, times=(1.0, 1.1), language="en"):
    outputs = tuple(
        [{"text": text, "language": language, "segments": []}] for text in texts
    )
    return SimpleNamespace(
        outputs=outputs, durations_seconds=times, median_seconds=sum(times) / 2
    )


def test_normalization_unicode_and_punctuation():
    assert normalize_text("  ＨＥＬＬＯ, World! ") == "hello world"


def test_reference_quality_substitution_insertion_and_silence():
    quality = text_quality("one two three", "one too three extra")
    assert quality["word_errors"] == 2
    assert quality["wer"] == pytest.approx(2 / 3)
    assert text_quality("", "")["wer"] == 0
    assert text_quality("", "hallucination")["empty_reference_hallucination"]
    assert text_quality("", "hallucination")["wer"] is None


def test_no_references_cannot_qualify_speed_claim():
    report = assess_backend_results(
        [result(["Hello", "Hello"]), result(["Hello", "Hello"], (2.0, 2.2))]
    )
    assert report["median_speedup"] == 2
    assert report["quality_status"] == "unqualified-no-references"
    assert not report["speed_claim_qualified"]


def test_relative_quality_gate_checks_all_repetitions():
    report = assess_backend_results(
        [result(["bad words", "hello world"]), result(["hello world", "hello world"])],
        ["hello world"],
    )
    assert report["quality_status"] == "failed"
    assert not report["speed_claim_qualified"]


def test_matching_reference_and_language_pass():
    report = assess_backend_results(
        [result(["Hello!", "Hello"]), result(["hello", "hello"], (2.0, 2.2))], ["hello"]
    )
    assert report["quality_status"] == "passed-relative-text-gate"
    assert report["speed_claim_qualified"]
    assert report["paired_round_speedups"] == [2.0, 2.0]


def test_language_mismatch_fails():
    report = assess_backend_results(
        [result(["Hi", "Hi"]), result(["Hi", "Hi"], language="de")], ["Hi"]
    )
    assert not report["speed_claim_qualified"]


@pytest.mark.parametrize("times", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timings_rejected(times):
    with pytest.raises(ValueError):
        summarize_times(times)


def test_bit_parallel_distance_matches_dynamic_programming():
    import random

    random.seed(7)
    for _ in range(100):
        left = [random.randrange(5) for _ in range(random.randrange(30))]
        right = [random.randrange(5) for _ in range(random.randrange(30))]
        row = list(range(len(right) + 1))
        for i, a in enumerate(left, 1):
            next_row = [i]
            for j, b in enumerate(right, 1):
                next_row.append(
                    min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b))
                )
            row = next_row
        assert edit_distance(left, right) == row[-1]


def test_per_clip_wall_ratios_include_parent_boundary():
    mlx, cpp = result(["hello", "hello"]), result(["hello", "hello"], (2.0, 2.2))
    for output in mlx.outputs:
        output[0].update(wall_seconds=1.0, service_seconds=0.9)
    for output in cpp.outputs:
        output[0].update(wall_seconds=2.0, service_seconds=1.5)
    clip = assess_backend_results([mlx, cpp])["per_clip"][0]
    assert clip["wall_median_speedup"] == 2
    assert clip["service_median_speedup"] == pytest.approx(1.5 / 0.9)
    assert clip["wall_times"]["whisper_cpp"]["samples_seconds"] == [2.0, 2.0]


def test_empty_reference_uses_hallucination_gate_even_when_cpp_is_worse():
    report = assess_backend_results([result(["", ""]), result(["you", "you"])], [""])
    assert report["quality_status"] == "passed-relative-text-gate"
    assert report["speed_claim_qualified"]
    assert report["per_clip"][0]["quality_pass"]
    assert report["per_clip"][0]["quality"]["whisper_cpp"][0]["wer"] is None


def test_both_engines_hallucinating_on_empty_reference_fails():
    report = assess_backend_results(
        [result(["you", "you"]), result(["you", "you"])], [""]
    )
    assert report["quality_status"] == "failed"
    assert not report["per_clip"][0]["quality_pass"]
