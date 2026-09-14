import mlx.core as mx
import pytest

from lightning_whisper_mlx.mtp import (
    MTPCheckpointMetadata,
    MTPHead,
    validate_checkpoint_metadata,
)


def test_mtp_head_returns_one_logit_matrix_per_future_horizon():
    head = MTPHead(hidden_size=8, horizons=2)
    hidden = mx.random.normal((1, 3, 8))
    embedding = mx.random.normal((32, 8))

    logits = head(hidden, lambda future_hidden: future_hidden @ embedding.T)

    assert logits.shape == (1, 3, 2, 32)


def test_mtp_checkpoint_metadata_rejects_incompatible_base():
    metadata = MTPCheckpointMetadata(
        base_model_repository="org/whisper",
        base_model_revision="abc123",
        base_model_fingerprint="fingerprint-a",
        decoder_hidden_size=8,
        vocabulary_size=32,
        tokenizer_fingerprint="tokenizer-a",
        horizons=2,
        weight_dtype="float32",
        training_config={"seed": 1},
        label_manifest_digest="labels-a",
        dataset_split_digest="split-a",
        training_commit="commit-a",
        validation_metrics={"h2_accuracy": 0.5},
    )

    validate_checkpoint_metadata(
        metadata,
        base_model_fingerprint="fingerprint-a",
        tokenizer_fingerprint="tokenizer-a",
        decoder_hidden_size=8,
        vocabulary_size=32,
    )

    with pytest.raises(ValueError, match="base model fingerprint"):
        validate_checkpoint_metadata(
            metadata,
            base_model_fingerprint="fingerprint-b",
            tokenizer_fingerprint="tokenizer-a",
            decoder_hidden_size=8,
            vocabulary_size=32,
        )

    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        validate_checkpoint_metadata(
            metadata,
            base_model_fingerprint="fingerprint-a",
            tokenizer_fingerprint="tokenizer-b",
            decoder_hidden_size=8,
            vocabulary_size=32,
        )
