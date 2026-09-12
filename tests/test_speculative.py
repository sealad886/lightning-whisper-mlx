import mlx.core as mx
import pytest

from lightning_whisper_mlx.speculative import (
    SpeculativeOptions,
    validate_speculative_options,
    verify_greedy_proposals,
)


def test_full_acceptance_commits_bonus_token():
    result = verify_greedy_proposals(
        prefix=[10],
        proposals=[1, 2],
        target_logits=[
            mx.array([0.0, 5.0, 0.0, 0.0]),
            mx.array([0.0, 0.0, 5.0, 0.0]),
            mx.array([0.0, 0.0, 0.0, 5.0]),
        ],
        eot=3,
    )

    assert result.tokens == [1, 2, 3]
    assert result.accepted == 2
    assert result.rejected == 0
    assert result.bonus_tokens == 1


def test_rejection_commits_target_correction_and_stops_block():
    result = verify_greedy_proposals(
        prefix=[10],
        proposals=[1, 2, 3],
        target_logits=[
            mx.array([0.0, 5.0, 0.0, 0.0]),
            mx.array([0.0, 0.0, 0.0, 5.0]),
            mx.array([0.0, 0.0, 5.0, 0.0]),
        ],
        eot=3,
    )

    assert result.tokens == [1, 3]
    assert result.accepted == 1
    assert result.rejected == 1
    assert result.bonus_tokens == 0


def test_acceleration_requires_greedy_decoding():
    with pytest.raises(ValueError, match="temperature=0"):
        validate_speculative_options(
            SpeculativeOptions(speculative_mode="assistant"), temperature=0.5
        )

    with pytest.raises(ValueError, match="lookahead"):
        validate_speculative_options(
            SpeculativeOptions(speculative_mode="mtp", lookahead=0), temperature=0.0
        )
