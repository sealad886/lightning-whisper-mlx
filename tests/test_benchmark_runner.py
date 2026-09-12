import pytest

from benchmarks.runner import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkSuite,
    run_benchmark,
)


def test_defaults_are_safe_and_explicit():
    config = BenchmarkConfig()

    assert config.warmups == 1
    assert config.repetitions == 3
    assert config.cases == (BenchmarkCase(mode="none"),)


def test_warmups_are_excluded_and_repetitions_are_recorded():
    calls = []

    def runner(case):
        calls.append(case.mode)
        return {"tokens": [1, 2], "token_count": 2}

    result = run_benchmark(runner, BenchmarkCase(mode="none"), warmups=2, repetitions=3)

    assert len(calls) == 5
    assert len(result.durations_seconds) == 3
    assert result.token_count == 2
    assert result.output["tokens"] == [1, 2]
    assert result.repetitions == 3


def test_suite_preserves_case_matrix_order():
    config = BenchmarkConfig(
        cases=(BenchmarkCase(mode="none"), BenchmarkCase(mode="assistant")),
        warmups=0,
        repetitions=1,
    )
    suite = BenchmarkSuite(config)

    results = suite.run(lambda case: {"token_count": len(case.mode)})

    assert [result.case.mode for result in results] == ["none", "assistant"]
    assert [result.token_count for result in results] == [4, 9]


def test_counts_must_be_nonnegative_and_repetitions_positive():
    with pytest.raises(ValueError, match="warmups"):
        BenchmarkConfig(warmups=-1)
    with pytest.raises(ValueError, match="repetitions"):
        BenchmarkConfig(repetitions=0)
