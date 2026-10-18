"""Policy scheduling contracts exercise CPU callables without importing MLX."""

import pytest

from benchmarks.profile_inference import POLICIES, compare_policies


def test_policy_comparison_interleaves_rotates_and_keeps_resource_samples():
    calls = []
    before = []
    after = []

    def runner(policy):
        calls.append(policy)
        return {"text": "same transcript", "tokens": [1, 2]}

    def before_sample():
        before.append(len(calls))

    def after_sample():
        after.append(len(calls))
        return {"active_bytes": len(calls), "scope": "shared process sample"}

    payload = compare_policies(
        runner,
        warmups=1,
        repetitions=2,
        before_sample=before_sample,
        after_sample=after_sample,
    )
    count = len(POLICIES)
    assert len(calls) == count * 3
    assert all(calls.count(policy) == 3 for policy in POLICIES)
    assert len(before) == len(after) == count * 2
    # Warmup consumes no resource hooks; each timed round changes starting case.
    assert calls[:count] == list(POLICIES)
    assert calls[count : 2 * count] == list(POLICIES)
    assert calls[2 * count :] == list(POLICIES[1:] + POLICIES[:1])
    assert payload["config"]["execution_policy"] == "interleaved-rotating-start"
    assert payload["text_outputs_equal"]
    assert all(len(result["durations_seconds"]) == 2 for result in payload["results"])
    assert all(len(result["sample_metrics"]) == 2 for result in payload["results"])
    assert all(
        metric["scope"] == "shared process sample"
        for result in payload["results"]
        for metric in result["sample_metrics"]
    )
    # Decoder-only compilation must be selectable independently of encoder/pipeline.
    assert any(
        not encoder and decoder and not pipeline
        for _, encoder, decoder, pipeline in POLICIES
    )


def test_policy_comparison_reports_changed_observable_output():
    payload = compare_policies(
        lambda policy: {"text": policy[0]}, warmups=0, repetitions=1
    )
    assert not payload["text_outputs_equal"]


def test_policy_comparison_selects_joint_generation_separately():
    payload = compare_policies(
        lambda policy: {"text": "same"}, warmups=0, repetitions=1
    )
    by_name = {policy["name"]: policy for policy in payload["policies"]}
    assert by_name["compiled_joint"]["joint_decoding"]
    assert by_name["compiled_joint"]["compile_decoder"]
    assert not by_name["compiled_joint"]["pipeline"]
    assert not by_name["compiled_both"]["joint_decoding"]
    assert by_name["compiled_joint_2"]["decode_chunk_size"] == 2
    assert by_name["compiled_joint_4"]["decode_chunk_size"] == 4


def test_policy_comparison_rejects_invalid_execution_counts():
    with pytest.raises(ValueError, match="repetitions"):
        compare_policies(lambda policy: {"text": ""}, repetitions=0)
