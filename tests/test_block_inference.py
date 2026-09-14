import mlx.core as mx

from lightning_whisper_mlx.decoding import Inference
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper


def make_model():
    return Whisper(
        ModelDimensions(
            n_mels=4,
            n_audio_ctx=3,
            n_audio_state=8,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=32,
            n_text_ctx=16,
            n_text_state=8,
            n_text_head=2,
            n_text_layer=2,
        ),
        dtype=mx.float32,
    )


def test_block_logits_match_sequential_logits():
    model = make_model()
    audio_features = mx.random.normal((1, 3, 8))
    tokens = mx.array([[1, 2, 3, 4]])

    sequential = Inference(model, initial_token_length=1)
    sequential_logits = [
        sequential.prefill(tokens[:, :1], audio_features),
        sequential.forward_one(tokens[:, 1:2], audio_features),
        sequential.forward_one(tokens[:, 2:3], audio_features),
        sequential.forward_one(tokens[:, 3:4], audio_features),
    ]

    block = Inference(model, initial_token_length=1)
    block_logits = [
        block.prefill(tokens[:, :1], audio_features),
        block.forward_block(tokens[:, 1:], audio_features),
    ]

    expected = mx.concatenate([sequential_logits[0], *sequential_logits[1:]], axis=1)
    actual = mx.concatenate(block_logits, axis=1)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


def test_baseline_logits_match_full_history_across_decode_steps():
    model = make_model()
    audio_features = mx.random.normal((1, 3, 8))
    tokens = mx.array([[1, 2, 3, 4]])
    inference = Inference(model, initial_token_length=2)

    for length in range(2, tokens.shape[-1] + 1):
        history = tokens[:, :length]
        actual = inference.logits(history, audio_features)
        expected = model.logits(history, audio_features)
        if length > 2:
            expected = expected[:, -1:]

        assert actual.dtype == mx.float32
        assert actual.shape == expected.shape
        assert inference.cache_length == length
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


def test_truncate_and_replay_matches_original_forward():
    model = make_model()
    audio_features = mx.random.normal((1, 3, 8))
    tokens = mx.array([[1, 2, 3]])
    inference = Inference(model, initial_token_length=1)

    inference.prefill(tokens[:, :1], audio_features)
    original = inference.forward_block(tokens[:, 1:2], audio_features)
    assert inference.cache_length == 2

    inference.truncate(1)
    assert inference.cache_length == 1
    replayed = inference.forward_one(tokens[:, 1:2], audio_features)

    assert mx.allclose(replayed, original, atol=1e-5, rtol=1e-5).item()
