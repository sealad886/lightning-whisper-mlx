"""Prompt/prefix composition preserves recent context within decoder capacity."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from lightning_whisper_mlx.decoding import DecodingOptions, DecodingTask, decode
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def task_with_context(*, prefix, prompt=None, sample_len=None):
    model = SimpleNamespace(
        dims=SimpleNamespace(
            n_text_ctx=32, n_vocab=51865, n_audio_ctx=3, n_audio_state=8
        ),
        is_multilingual=True,
        num_languages=99,
    )
    return DecodingTask(
        model,
        DecodingOptions(
            language="en",
            fp16=False,
            without_timestamps=True,
            prefix=prefix,
            prompt=prompt,
            sample_len=sample_len,
        ),
    )


@pytest.mark.parametrize("sample_len", [None, 1, 25])
def test_valid_composition_preserves_prefix_and_historical_prompt(sample_len):
    prefix, prompt = list(range(100, 110)), list(range(200, 204))
    task = task_with_context(prefix=prefix, prompt=prompt, sample_len=sample_len)
    assert task.initial_tokens == (
        task.tokenizer.sot_prev,
        *prompt,
        *task.sot_sequence,
        *prefix,
    )


def test_overflow_truncates_old_prompt_before_current_prefix():
    prefix, prompt = list(range(100, 120)), list(range(200, 220))
    task = task_with_context(prefix=prefix, prompt=prompt)
    prompt_budget = task.n_ctx - 1 - len(task.sot_sequence) - len(prefix) - 1
    assert task.initial_tokens == (
        task.tokenizer.sot_prev,
        *prompt[-prompt_budget:],
        *task.sot_sequence,
        *prefix,
    )
    assert task.sample_begin == task.n_ctx - 1


@pytest.mark.parametrize("with_prompt", [False, True])
def test_oversized_prefix_retains_newest_tokens_and_reserves_generation(with_prompt):
    prefix = list(range(100, 150))
    task = task_with_context(prefix=prefix, prompt=[200] * 10 if with_prompt else None)
    prefix_budget = task.n_ctx - 1 - len(task.sot_sequence)
    assert task.initial_tokens == (*task.sot_sequence, *prefix[-prefix_budget:])
    assert task.sample_begin == task.n_ctx - 1


def test_string_prefix_uses_same_context_budget():
    prefix = " hello" * 100
    task = task_with_context(prefix=prefix)
    encoded = task.tokenizer.encode(" " + prefix.strip())
    prefix_budget = task.n_ctx - 1 - len(task.sot_sequence)
    assert task.initial_tokens == (*task.sot_sequence, *encoded[-prefix_budget:])


def test_public_decode_accepts_prefix_larger_than_text_context():
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 51865, 32, 8, 2, 1), mx.float32)
    result = decode(
        model,
        mx.zeros((3, 8)),
        DecodingOptions(
            language="en",
            fp16=False,
            without_timestamps=True,
            prefix=list(range(100, 150)),
            prompt=[200] * 10,
        ),
    )
    assert len(result.tokens) <= 2
    assert result.language == "en"
