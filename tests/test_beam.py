"""Beam selection, completion, and cache ownership behavior."""

import mlx.core as mx
import pytest

from lightning_whisper_mlx.beam import BeamSearchDecoder


class Inference:
    def __init__(self):
        self.parents = []

    def rearrange_kv_cache(self, parents):
        self.parents.append(list(parents))


def test_beam_preserves_a_better_non_greedy_sequence():
    inference = Inference()
    decoder = BeamSearchDecoder(2, 4, inference)
    tokens, _, scores = decoder.update(
        mx.array([[0], [0]]),
        mx.array([[-20, 3.0, 2.9, -20, -20]] * 2),
        mx.zeros(2),
    )
    assert tokens.tolist() == [[0, 1], [0, 2]]
    assert inference.parents[-1] == [0, 0]
    tokens, complete, scores = decoder.update(
        tokens,
        mx.array([[-20, -20, -20, 0.0, -0.1], [-20, -20, -20, 0.0, 5.0]]),
        scores,
    )
    assert complete
    candidates, probabilities = decoder.finalize(
        tokens.reshape(1, 2, -1), scores.reshape(1, 2)
    )
    best = max(range(len(probabilities[0])), key=probabilities[0].__getitem__)
    assert candidates[0, best].tolist()[:3] == [0, 2, 4]


def test_patience_retains_more_completed_candidates_before_stopping():
    decoder = BeamSearchDecoder(2, 4, Inference(), patience=2)
    tokens, complete, scores = decoder.update(
        mx.array([[0], [1]]), mx.array([[0, -1, -2, -3, 5]] * 2), mx.zeros(2)
    )
    assert not complete
    assert len(decoder.finished_sequences[0]) == 2
    tokens, complete, scores = decoder.update(
        tokens, mx.array([[0, -1, -2, -3, 5]] * 2), scores
    )
    assert complete
    assert len(decoder.finished_sequences[0]) == 4
    final, values = decoder.finalize(tokens.reshape(1, 2, -1), scores.reshape(1, 2))
    assert final.shape[1] == len(values[0]) == 4
    assert all(decoder.eot in sequence for sequence in final[0].tolist())


def test_audio_groups_never_share_hypotheses_or_cache_parents():
    inference = Inference()
    decoder = BeamSearchDecoder(2, 4, inference)
    tokens, _, scores = decoder.update(
        mx.array([[10], [10], [20], [20]]),
        mx.array([[0, 5, 4, -20, -20]] * 4),
        mx.array([0.0, -100.0, -100.0, 0.0]),
    )
    assert inference.parents[-1] == [0, 0, 3, 3]
    assert tokens.tolist() == [[10, 1], [10, 2], [20, 1], [20, 2]]
    final, values = decoder.finalize(tokens.reshape(2, 2, -1), scores.reshape(2, 2))
    assert all(sequence[0] == 10 for sequence in final[0].tolist())
    assert all(sequence[0] == 20 for sequence in final[1].tolist())
    assert len(values) == 2


def test_equal_scores_choose_lexicographic_tokens_deterministically():
    decoder = BeamSearchDecoder(2, 5, Inference())
    tokens, _, _ = decoder.update(mx.array([[9], [9]]), mx.zeros((2, 6)), mx.zeros(2))
    assert tokens.tolist() == [[9, 0], [9, 1]]


def test_exhausted_beam_completes_with_available_finished_hypotheses():
    decoder = BeamSearchDecoder(2, 2, Inference(), patience=3)
    tokens, completed, scores = decoder.update(
        mx.array([[0], [0]]),
        mx.array([[-float("inf"), -float("inf"), 0.0]] * 2),
        mx.zeros(2),
    )
    assert completed
    final, values = decoder.finalize(tokens.reshape(1, 2, -1), scores.reshape(1, 2))
    assert final.tolist() == [[[0, 2]]]
    assert values == [[0.0]]


def test_finalization_pads_candidate_groups_without_changing_live_scores():
    decoder = BeamSearchDecoder(2, 4, Inference(), patience=2)
    decoder.finished_sequences = [
        {(0, 1, 4): -0.1, (0, 2, 4): -0.2, (0, 3, 4): -0.3},
        {(0, 1, 4): -0.4},
    ]
    tokens = mx.array([[[0, 1], [0, 2]], [[0, 2], [0, 3]]])
    scores = mx.array([[-1.0, -2.0], [-1.0, -2.0]])
    final, values = decoder.finalize(tokens, scores)
    assert final.shape == (2, 3, 3)
    assert values[0] == [-0.1, -0.2, -0.3]
    assert values[1][:2] == [-0.4, -1.0]
    assert values[1][2] == -float("inf")
    assert scores.tolist() == [[-1.0, -2.0], [-1.0, -2.0]]


def test_reset_discards_previous_audio_hypotheses():
    decoder = BeamSearchDecoder(1, 2, Inference())
    decoder.update(mx.array([[0]]), mx.array([[-5, -5, 5.0]]), mx.zeros(1))
    assert decoder.finished_sequences[0]
    decoder.reset()
    assert decoder.finished_sequences is None
    tokens, _, _ = decoder.update(
        mx.array([[1]]), mx.array([[5.0, -5, -5]]), mx.zeros(1)
    )
    assert tokens.tolist() == [[1, 0]]


@pytest.mark.parametrize(
    "beam, eot, patience",
    [
        (0, 1, None),
        (True, 1, None),
        (1.5, 1, None),
        (1, -1, None),
        (1, True, None),
        (1, 1, 0),
        (1, 1, float("inf")),
        (1, 1, float("nan")),
        (1, 1, 0.1),
    ],
)
def test_invalid_parameters_fail_clearly(beam, eot, patience):
    with pytest.raises(ValueError):
        BeamSearchDecoder(beam, eot, Inference(), patience)


def test_invalid_rows_and_fully_masked_vocabulary_fail():
    decoder = BeamSearchDecoder(2, 2, Inference())
    with pytest.raises(ValueError, match="divisible"):
        decoder.update(mx.array([[0]]), mx.zeros((1, 3)), mx.zeros(1))
    with pytest.raises(ValueError, match="finite continuation"):
        decoder.update(
            mx.array([[0], [0]]), mx.full((2, 3), -float("inf")), mx.zeros(2)
        )


def test_public_decode_uses_beam_search_for_independent_audio(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from lightning_whisper_mlx import decoding

    model = SimpleNamespace(
        dims=SimpleNamespace(
            n_text_ctx=16, n_vocab=51865, n_audio_ctx=3, n_audio_state=8
        ),
        is_multilingual=True,
        num_languages=99,
    )
    tokenizer = decoding.get_tokenizer(
        True, num_languages=99, language="en", task="transcribe"
    )
    parent_maps = []

    class FixedInference:
        def __init__(self, model, initial_length):
            self.initial_length = initial_length

        def logits(self, tokens, features):
            values = np.full(
                (tokens.shape[0], tokens.shape[1], 51865), -30.0, dtype=np.float32
            )
            for row, history in enumerate(tokens.tolist()):
                first = 10 if features[row, 0, 0].item() == 0 else 20
                if len(history) == self.initial_length:
                    values[row, -1, first] = 3.0
                    values[row, -1, first + 1] = 2.9
                elif history[-1] == first:
                    values[row, -1, first + 2] = 0.0
                    values[row, -1, tokenizer.eot] = -0.1
                else:
                    values[row, -1, first + 2] = 0.0
                    values[row, -1, tokenizer.eot] = 5.0
            return mx.array(values)

        def rearrange_kv_cache(self, parents):
            parent_maps.append(parents)

        def reset(self):
            pass

    monkeypatch.setattr(decoding, "Inference", FixedInference)
    features = mx.stack([mx.zeros((3, 8)), mx.ones((3, 8))])
    results = decoding.decode(
        model,
        features,
        decoding.DecodingOptions(
            language="en",
            beam_size=2,
            fp16=False,
            without_timestamps=True,
            suppress_blank=False,
            suppress_tokens=[],
            sample_len=2,
        ),
    )
    assert [result.tokens for result in results] == [[11], [21]]
    assert parent_maps[0] == [0, 0, 2, 2]
    assert [result.audio_features[0, 0].item() for result in results] == [0, 1]
