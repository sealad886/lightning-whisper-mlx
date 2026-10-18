"""Compiled sampling shares eager filters, token history, and EOT scoring."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx.decoding import (
    ApplyTimestampRules,
    GreedyDecoder,
    SuppressBlank,
    SuppressTokens,
    _compiled_greedy_sampler,
)


def tokenizer():
    return SimpleNamespace(
        eot=10, no_timestamps=19, timestamp_begin=20, encode=lambda text: [1]
    )


def filters(timestamps=True, max_initial=0):
    output = [SuppressBlank(tokenizer(), 1, 32), SuppressTokens([3, 4], 32)]
    if timestamps:
        output.append(ApplyTimestampRules(tokenizer(), 1, max_initial))
    return output


@pytest.mark.parametrize(
    "history",
    [[9], [9, 20], [9, 20, 2], [9, 20, 2, 24], [9, 20, 24, 2], [9, 25, 1, 22]],
)
def test_timestamp_rules_ignore_unused_fixed_buffer_positions(history):
    rule = ApplyTimestampRules(tokenizer(), 1, 0)
    compact = mx.array([history])
    buffer = mx.array([history + [31] * (16 - len(history))])
    logits = mx.array(np.linspace(-2, 2, 32, dtype=np.float32)[None, :])
    expected = rule.apply(logits, compact)
    actual = rule.apply(logits, buffer, logical_length=mx.array(len(history)))
    assert np.array_equal(
        np.isneginf(np.array(actual)), np.isneginf(np.array(expected))
    )
    finite = np.isfinite(np.array(expected))
    assert np.allclose(np.array(actual)[finite], np.array(expected)[finite])
    assert np.isfinite(np.array(actual)).any()


@pytest.mark.parametrize(
    "history", [[9], [9, 20], [9, 20, 2], [9, 20, 2, 24], [9, 20, 24, 2]]
)
@pytest.mark.parametrize("timestamps", [False, True])
def test_compiled_sampler_matches_filtered_eager_tokens_and_scores(history, timestamps):
    rules = filters(timestamps)
    eager = GreedyDecoder(0, 10, 32)
    compiled = GreedyDecoder(0, 10, 32)
    compiled.configure_sampling(rules)
    first = mx.array([history, history])
    second = first
    eager_scores = mx.array([-0.3, -0.7])
    compiled_scores = eager_scores
    for step in range(4):
        raw = mx.array(
            np.random.default_rng(step).normal(size=(2, 32)).astype(np.float32)
        )
        filtered = raw
        for rule in rules:
            filtered = rule.apply(filtered, first)
        first, done_eager, eager_scores = eager.update(first, filtered, eager_scores)
        second, done_compiled, compiled_scores = compiled.update(
            second, raw, compiled_scores
        )
        assert first.tolist() == second.tolist()
        assert done_eager.item() == done_compiled.item()
        assert mx.allclose(eager_scores, compiled_scores, atol=1e-5, rtol=1e-5).item()
        assert compiled._tokens.shape == (2, 32)


def test_eot_rows_keep_scores_even_when_all_logits_are_masked():
    eager = GreedyDecoder(0, 10, 8)
    compiled = GreedyDecoder(0, 10, 8)
    compiled.configure_sampling([SuppressTokens([3], 32)])
    tokens = mx.array([[9], [10]])
    logits = mx.full((2, 32), -float("inf"))
    logits[0, 2] = 1.0
    scores = mx.array([-0.1, -0.7])
    first, _, first_scores = eager.update(tokens, logits, scores)
    second, _, second_scores = compiled.update(tokens, logits, scores)
    assert first.tolist() == second.tolist() == [[9, 2], [10, 10]]
    assert np.isfinite(np.array(second_scores)).all()
    assert mx.allclose(first_scores, second_scores).item()
    assert second_scores[1].item() == pytest.approx(-0.7)


def test_blank_suppression_changes_only_at_initial_boundary():
    rule = SuppressBlank(tokenizer(), 1, 32)
    logits = mx.zeros((1, 32))
    tokens = mx.array([[9, 1, 1, 1]])
    initial = rule.apply(logits, tokens, logical_length=mx.array(1))
    later = rule.apply(logits, tokens, logical_length=mx.array(2))
    assert np.isneginf(initial[0, 1].item())
    assert np.isneginf(initial[0, 10].item())
    assert later[0, 1].item() == later[0, 10].item() == 0


def test_reset_reuses_compiled_graph_without_previous_token_history():
    rules = filters(False)
    decoder = GreedyDecoder(0, 10, 8)
    decoder.configure_sampling(rules)
    sampler = decoder._compiled_sampler
    decoder.update(mx.array([[9]]), mx.zeros((1, 32)), mx.zeros(1))
    decoder.reset()
    assert decoder._tokens is None
    result, _, _ = decoder.update(mx.array([[8, 7]]), mx.zeros((1, 32)), mx.zeros(1))
    assert result.tolist() == [[8, 7, 0]]
    assert decoder._compiled_sampler is sampler
    same_configuration = GreedyDecoder(0, 10, 8)
    same_configuration.configure_sampling(filters(False))
    assert same_configuration._compiled_sampler is sampler
    assert _compiled_greedy_sampler.cache_info().maxsize == 16


@pytest.mark.parametrize("history", [[], [1, 2, 3, 4]])
def test_fixed_capacity_rejects_empty_or_overflowing_history(history):
    decoder = GreedyDecoder(0, 10, 4)
    decoder.configure_sampling([])
    with pytest.raises(ValueError, match="history"):
        decoder.update(mx.array([history]), mx.zeros((1, 32)), mx.zeros(1))


def test_history_length_change_requires_reset():
    decoder = GreedyDecoder(0, 10, 8)
    decoder.configure_sampling([])
    decoder.update(mx.array([[1]]), mx.zeros((1, 32)), mx.zeros(1))
    with pytest.raises(ValueError, match="length changed"):
        decoder.update(mx.array([[1]]), mx.zeros((1, 32)), mx.zeros(1))


@pytest.mark.parametrize("length", [0, 5])
def test_explicit_invalid_logical_length_is_rejected(length):
    rule = ApplyTimestampRules(tokenizer(), 1, 0)
    with pytest.raises(ValueError, match="logical token length"):
        rule.apply(mx.zeros((1, 32)), mx.array([[9, 20, 1, 22]]), logical_length=length)
