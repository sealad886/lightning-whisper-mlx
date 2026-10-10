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


def test_explicit_token_count_overrides_token_length_in_both_public_apis():
    calls = {"none": 0, "assistant": 0}

    def runner(case):
        calls[case.mode] += 1
        return {"tokens": [1, 2], "token_count": 10 + calls[case.mode]}

    result = run_benchmark(runner, BenchmarkCase(), warmups=0, repetitions=2)
    assert result.token_count == 12
    assert [output["token_count"] for output in result.outputs] == [11, 12]
    calls = {"none": 0, "assistant": 0}
    suite = BenchmarkSuite(
        BenchmarkConfig(
            cases=(BenchmarkCase(), BenchmarkCase(mode="assistant")),
            warmups=0,
            repetitions=2,
        )
    )
    results = suite.run(runner)
    assert [result.token_count for result in results] == [12, 12]
    assert all(
        [output["token_count"] for output in result.outputs] == [11, 12]
        for result in results
    )


def test_suite_balances_case_positions_and_retains_round_order():
    cases = tuple(BenchmarkCase(mode=mode) for mode in ("none", "assistant", "mtp"))
    suite = BenchmarkSuite(BenchmarkConfig(cases=cases, warmups=1, repetitions=3))
    shared = {"tokens": []}

    def runner(case):
        shared["tokens"][:] = [case.mode]
        return shared

    results = suite.run(runner)
    payload = suite.to_dict(results)
    timed = [entry for entry in payload["execution_order"] if entry["phase"] == "timed"]

    assert len(payload["execution_order"]) == 12
    for round_index in range(3):
        assert {
            entry["case_index"]
            for entry in timed
            if entry["round_index"] == round_index
        } == set(range(3))
    for case_index, result in enumerate(results):
        assert {entry["position"] for entry in result.sample_order} == set(range(3))
        assert all(entry["case_index"] == case_index for entry in result.sample_order)
        assert all(value["tokens"] == [result.case.mode] for value in result.outputs)
        assert [
            {key: value for key, value in entry.items() if key != "phase"}
            for entry in timed
            if entry["case_index"] == case_index
        ] == list(result.sample_order)


def test_timing_excludes_warmups_and_resource_hooks(monkeypatch):
    from benchmarks import runner as benchmark_runner

    clock = [0.0]
    monkeypatch.setattr(benchmark_runner.time, "perf_counter", lambda: clock[0])
    counts = {"none": 0, "assistant": 0}
    resource_record = {}

    def runner(case):
        counts[case.mode] += 1
        clock[0] += 100 if counts[case.mode] == 1 else len(case.mode)
        return {"tokens": [counts[case.mode]]}

    def before_sample():
        clock[0] += 1000

    def after_sample():
        clock[0] += 1000
        resource_record["completed_calls"] = sum(counts.values())
        return resource_record

    suite = BenchmarkSuite(
        BenchmarkConfig(
            cases=(BenchmarkCase(), BenchmarkCase(mode="assistant")),
            warmups=1,
            repetitions=2,
        )
    )
    results = suite.run(runner, before_sample=before_sample, after_sample=after_sample)

    assert [result.durations_seconds for result in results] == [(4.0, 4.0), (9.0, 9.0)]
    assert counts == {"none": 3, "assistant": 3}
    measured = sorted(
        (order["execution_index"], sample["completed_calls"])
        for result in results
        for order, sample in zip(result.sample_order, result.sample_metrics)
    )
    assert [count for _, count in measured] == [3, 4, 5, 6]


def test_counts_must_be_nonnegative_and_repetitions_positive():
    with pytest.raises(ValueError, match="warmups"):
        BenchmarkConfig(warmups=-1)
    with pytest.raises(ValueError, match="repetitions"):
        BenchmarkConfig(repetitions=0)


def test_each_repetition_preserves_its_output_even_when_runner_reuses_mapping():
    output = {"tokens": []}

    def runner(case):
        output["tokens"].append(len(output["tokens"]))
        return output

    result = run_benchmark(runner, BenchmarkCase(), warmups=0, repetitions=3)

    assert [value["tokens"] for value in result.outputs] == [[0], [0, 1], [0, 1, 2]]


def test_caller_metadata_cannot_replace_measured_provenance():
    result = run_benchmark(
        lambda case: None,
        BenchmarkCase(),
        warmups=0,
        repetitions=1,
        metadata={"git_revision": "invented", "fixture": "speech"},
    )

    assert result.metadata["git_revision"] != "invented"
    assert result.metadata["fixture"] == "speech"
    assert result.metadata["mlx_version"] not in {"unknown", ""}
