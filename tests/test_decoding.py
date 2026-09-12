import mlx.core as mx

from lightning_whisper_mlx.decoding import GreedyDecoder


def test_temperature_zero_selects_target_argmax():
    decoder = GreedyDecoder(temperature=0.0, eot=99)
    tokens, completed, logprobs = decoder.update(
        mx.array([[1, 2]]),
        mx.array([[0.1, 0.2, 4.0]]),
        mx.array([0.0]),
    )

    assert tokens.tolist() == [[1, 2, 2]]
    assert not completed.item()
    assert logprobs.shape == (1,)


def test_nonzero_temperature_uses_sampler(monkeypatch):
    sampled = mx.array([1])

    def fake_categorical(*, logits):
        assert logits.shape == (1, 3)
        return sampled

    monkeypatch.setattr(
        "lightning_whisper_mlx.decoding.mx.random.categorical", fake_categorical
    )

    decoder = GreedyDecoder(temperature=1.0, eot=99)
    tokens, completed, _ = decoder.update(
        mx.array([[1, 2]]),
        mx.array([[0.1, 0.2, 4.0]]),
        mx.array([0.0]),
    )

    assert tokens.tolist() == [[1, 2, 1]]
    assert not completed.item()
