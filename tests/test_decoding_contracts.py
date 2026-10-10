"""Decoder contracts for timestamp legality and sampling candidate groups."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx.decoding import (
    ApplyTimestampRules,
    DecodingOptions,
    DecodingTask,
    decode,
)


@pytest.mark.parametrize(
    "history,forbidden",
    [([9, 20, 1, 27], range(20, 27)), ([9, 20, 1, 27, 27, 2], range(20, 28))],
)
def test_timestamps_cannot_decrease_or_end_with_zero_duration(history, forbidden):
    rule = ApplyTimestampRules(
        SimpleNamespace(timestamp_begin=20, eot=10, no_timestamps=19), 1, None
    )
    filtered = rule.apply(mx.zeros((1, 32)), mx.array([history]))
    assert all(np.isneginf(filtered[0, token].item()) for token in forbidden)
    assert not np.all(np.isneginf(np.array(filtered)))


def test_forbidden_timestamp_mass_cannot_eliminate_valid_text():
    rule = ApplyTimestampRules(
        SimpleNamespace(timestamp_begin=20, eot=10, no_timestamps=19), 1, None
    )
    raw = mx.array([[-10.0] * 24 + [10.0] + [-10.0] * 7])
    filtered = rule.apply(raw, mx.array([[9, 20, 22]]))
    assert np.isfinite(filtered[0, 1].item())
    assert np.all(np.isneginf(np.array(filtered)[0, 20:]))


def test_zero_initial_timestamp_is_enforced():
    rule = ApplyTimestampRules(
        SimpleNamespace(timestamp_begin=20, eot=10, no_timestamps=19), 1, 0
    )
    raw = mx.array([[0.0] * 21 + [10.0] * 11])
    assert int(mx.argmax(rule.apply(raw, mx.array([[9]])), axis=-1)[0].item()) == 20


def test_best_of_groups_keep_each_audio_and_choose_best_candidate(monkeypatch):
    model = SimpleNamespace(
        dims=SimpleNamespace(
            n_text_ctx=16, n_vocab=51865, n_audio_ctx=3, n_audio_state=8
        ),
        is_multilingual=True,
        num_languages=99,
    )
    audio = mx.stack([mx.zeros((3, 8)), mx.ones((3, 8))])

    def sample(self, features, tokens, assistant_audio_features=None):
        assert features.shape[0] == tokens.shape[0] == 4
        assert features[:, 0, 0].tolist() == [0, 0, 1, 1]
        generated = mx.array([[1], [2], [3], [4]])
        return (
            mx.concatenate([tokens, generated], axis=-1),
            mx.array([-2.0, -1.0, -1.0, -2.0]),
            [0.1] * 4,
        )

    monkeypatch.setattr(DecodingTask, "_main_loop", sample)
    result = decode(
        model,
        audio,
        DecodingOptions(
            language="en",
            temperature=1,
            best_of=2,
            fp16=False,
            without_timestamps=True,
            sample_len=1,
        ),
    )
    assert [row.tokens for row in result] == [[2], [3]]
    assert [row.audio_features[0, 0].item() for row in result] == [0, 1]
