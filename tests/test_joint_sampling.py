"""Joint execution preserves ordinary decoding's observable numerical contract."""

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from lightning_whisper_mlx.decoding import (
    DecodingOptions,
    GreedyDecoder,
    Inference,
    SuppressTokens,
    _compiled_greedy_sampler,
    decode,
)
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


@pytest.mark.parametrize("timestamps", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_joint_generation_matches_separate_compiled_generation(
    timestamps, chunk_size, dtype
):
    mx.random.seed(17)
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 51865, 80, 8, 2, 1), mx.float32)
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    features = mx.random.normal((2, 3, 8)).astype(dtype)
    options = DecodingOptions(
        language="en",
        fp16=dtype == mx.float16,
        sample_len=8,
        without_timestamps=not timestamps,
        pipeline=False,
    )
    model.configure_execution(compile_decoder=True, joint_decoding=False)
    expected = decode(model, features, options)
    model.configure_execution(
        compile_decoder=True, joint_decoding=True, decode_chunk_size=chunk_size
    )
    actual = decode(model, features, options)
    assert [result.tokens for result in actual] == [
        result.tokens for result in expected
    ]
    assert [result.text for result in actual] == [result.text for result in expected]
    for left, right in zip(actual, expected):
        assert left.language == right.language
        assert left.no_speech_prob == right.no_speech_prob
        assert left.avg_logprob == pytest.approx(right.avg_logprob, abs=1e-5)


@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("sample_len", [1, 2, 3, 7, 24])
def test_bounded_joint_chunks_preserve_sample_and_context_tails(chunk_size, sample_len):
    mx.random.seed(19)
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 51865, 16, 8, 2, 1), mx.float32)
    features = mx.random.normal((2, 3, 8))
    options = DecodingOptions(
        language="en",
        fp16=False,
        sample_len=sample_len,
        without_timestamps=True,
        pipeline=False,
    )
    model.configure_execution(compile_decoder=True, joint_decoding=True)
    expected = decode(model, features, options)
    model.configure_execution(
        compile_decoder=True, joint_decoding=True, decode_chunk_size=chunk_size
    )
    actual = decode(model, features, options)
    assert [result.tokens for result in actual] == [
        result.tokens for result in expected
    ]
    for left, right in zip(actual, expected):
        assert left.avg_logprob == pytest.approx(right.avg_logprob, abs=1e-5)
        assert left.no_speech_prob == right.no_speech_prob


def test_joint_step_preserves_growth_rollback_and_reordered_cache_rows():
    mx.random.seed(23)
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 32, 128, 8, 2, 1), mx.float32)
    model.configure_execution(compile_decoder=True, joint_decoding=True)
    features = mx.random.normal((2, 3, 8))
    prefix = mx.array([[1] * 63, [2] * 63])
    first, joint = Inference(model, 63), Inference(model, 63)
    rules = [SuppressTokens([31], 32)]

    def sampler():
        result = GreedyDecoder(0, 31, 129)
        result.configure_sampling(rules)
        return result

    ordinary_sampler, joint_sampler = sampler(), sampler()
    left, _, left_scores = ordinary_sampler.update(
        prefix, first.prefill(prefix, features)[:, -1], mx.zeros(2)
    )
    right, _, right_scores = joint_sampler.update(
        prefix, joint.prefill(prefix, features)[:, -1], mx.zeros(2)
    )
    for _ in range(3):
        previous_length = right.shape[-1]
        right, _, right_scores = joint.sample_next(
            right, features, joint_sampler, right_scores, max_steps=4
        )
        for _ in range(right.shape[-1] - previous_length):
            left, _, left_scores = ordinary_sampler.update(
                left, first.logits(left, features)[:, -1], left_scores
            )
        assert left.tolist() == right.tolist()
        assert mx.allclose(left_scores, right_scores, atol=1e-5).item()
        assert first.cache_length == joint.cache_length
    assert joint.kv_cache[0][0].keys.shape[1] == 128
    first.truncate(64)
    joint.truncate(64)
    first.rearrange_kv_cache([1, 0])
    joint.rearrange_kv_cache([1, 0])
    left, right = left[[1, 0], :65], right[[1, 0], :65]
    features = features[[1, 0]]
    ordinary_sampler, joint_sampler = sampler(), sampler()
    left, _, left_scores = ordinary_sampler.update(
        left, first.logits(left, features)[:, -1], mx.zeros(2)
    )
    right, _, right_scores = joint.sample_next(
        right, features, joint_sampler, mx.zeros(2)
    )
    assert left.tolist() == right.tolist()
    assert mx.allclose(left_scores, right_scores, atol=1e-5).item()
    for (owner, cross), (joint_owner, joint_cross) in zip(
        first.kv_cache, joint.kv_cache
    ):
        assert owner.offset == joint_owner.offset == 65
        assert mx.allclose(
            owner.keys[:, :65], joint_owner.keys[:, :65], atol=1e-5
        ).item()
        assert mx.allclose(
            owner.values[:, :65], joint_owner.values[:, :65], atol=1e-5
        ).item()
        assert all(mx.array_equal(a, b).item() for a, b in zip(cross, joint_cross))


def test_joint_callable_cache_is_model_owned_bounded_and_invalidated(
    monkeypatch, request
):
    # Callable lifetime/capacity is the contract here; numerical compilation is
    # covered above. Avoid compiling eighteen equivalent tiny model graphs.
    request.addfinalizer(_compiled_greedy_sampler.cache_clear)
    monkeypatch.setattr(mx, "compile", lambda function: function)
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 32, 16, 8, 2, 1), mx.float32)
    other = Whisper(model.dims, mx.float32)
    model.configure_execution(compile_decoder=True, joint_decoding=True)
    other.configure_execution(compile_decoder=True, joint_decoding=True)
    features, prefix = mx.zeros((1, 3, 8)), mx.array([[1]])
    for excluded in range(18):
        inference = Inference(model, 1)
        sampler = GreedyDecoder(0, 31, 17)
        sampler.configure_sampling([SuppressTokens([excluded, 31], 32)])
        tokens, _, scores = sampler.update(
            prefix, inference.prefill(prefix, features)[:, -1], mx.zeros(1)
        )
        tokens, _, scores = inference.sample_next(tokens, features, sampler, scores)
        mx.eval(tokens, scores, inference.cache_state)
    assert len(model._joint_decoder_cache) == 16
    assert not other._joint_decoder_cache
    model.configure_execution(compile_decoder=True, joint_decoding=False)
    assert not model._joint_decoder_cache
    assert not model._joint_decoding


@pytest.mark.parametrize("chunk_size", [2, 4])
def test_joint_chunk_masks_divergent_eot_rows_without_extra_scores(chunk_size):
    model = Whisper(ModelDimensions(4, 3, 8, 2, 1, 51865, 16, 8, 2, 1), mx.float32)
    from lightning_whisper_mlx.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(True, num_languages=99, language="en", task="transcribe")
    prefix_length = len(tokenizer.sot_sequence_including_notimestamps)
    original = model.decoder

    def controlled(tokens, features, **kwargs):
        outputs = original(tokens, features, **kwargs)
        offset = kwargs.get("offset")
        generation = mx.array(0) if offset is None else offset - prefix_length + 1
        targets = mx.where(
            mx.arange(tokens.shape[0]) == 0,
            tokenizer.eot,
            mx.where(generation >= 3, tokenizer.eot, 10),
        )
        logits = mx.where(
            mx.arange(51865)[None, None] == targets[:, None, None], 5.0, -5.0
        )
        return (mx.broadcast_to(logits, outputs[0].shape), *outputs[1:])

    model.decoder = controlled
    features = mx.zeros((2, 3, 8))
    options = DecodingOptions(
        language="en",
        fp16=False,
        sample_len=10,
        without_timestamps=True,
        suppress_blank=False,
        suppress_tokens=[],
        pipeline=False,
    )
    model.configure_execution(compile_decoder=True, joint_decoding=True)
    expected = decode(model, features, options)
    model.configure_execution(
        compile_decoder=True, joint_decoding=True, decode_chunk_size=chunk_size
    )
    actual = decode(model, features, options)
    assert [result.tokens for result in actual] == [[], [10, 10, 10]]
    for left, right in zip(actual, expected):
        assert left.tokens == right.tokens
        assert left.avg_logprob == pytest.approx(right.avg_logprob, abs=1e-5)
        assert left.no_speech_prob == right.no_speech_prob
