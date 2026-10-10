import mlx.core as mx
import pytest
import mlx.nn as nn

from lightning_whisper_mlx.mtp import (
    MTPCheckpointMetadata,
    MTPHead,
    load_mtp_head,
    validate_checkpoint_metadata,
)
from training.train_mtp import MTPTrainer


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


@pytest.mark.parametrize("damage", ["missing", "unexpected", "wrong_shape"])
def test_mtp_checkpoint_rejects_incomplete_or_incompatible_weights(tmp_path, damage):
    trainer = MTPTrainer(
        hidden_size=8,
        vocabulary_size=32,
        horizons=2,
        learning_rate=0.01,
        output_projection=nn.Embedding(32, 8).as_linear,
    )
    checkpoint = tmp_path / "mtp"
    trainer.save_checkpoint(
        checkpoint,
        base_model_repository="org/model",
        base_model_revision="rev",
        base_model_fingerprint="base",
        tokenizer_fingerprint="tokenizer",
        label_manifest_digest="labels",
        dataset_split_digest="split",
        training_commit="commit",
    )
    weights_path = str(checkpoint / "weights.npz")
    weights = mx.load(weights_path)
    name = next(iter(weights))
    if damage == "missing":
        weights.pop(name)
    elif damage == "unexpected":
        weights["unexpected.weight"] = mx.zeros((8,))
    else:
        weights[name] = mx.zeros((1,))
    mx.savez(weights_path, **weights)

    with pytest.raises(ValueError):
        load_mtp_head(
            str(checkpoint),
            decoder_hidden_size=8,
            vocabulary_size=32,
            base_model_fingerprint="base",
            tokenizer_fingerprint="tokenizer",
        )


def test_tokenizer_fingerprint_identifies_full_vocabulary_and_segmentation():
    from types import SimpleNamespace
    import tiktoken
    from lightning_whisper_mlx.mtp import fingerprint_tokenizer

    def tokenizer(ranks, pattern=r"[\s\S]"):
        return SimpleNamespace(
            encoding=tiktoken.Encoding(
                name="fingerprint-test",
                pat_str=pattern,
                mergeable_ranks=ranks,
                special_tokens={"<|endoftext|>": 2},
            )
        )

    first = tokenizer({b"a": 0, b"b": 1})
    swapped = tokenizer({b"a": 1, b"b": 0})
    different_pattern = tokenizer({b"a": 0, b"b": 1}, r".+")
    assert fingerprint_tokenizer(first) != fingerprint_tokenizer(swapped)
    assert fingerprint_tokenizer(first) != fingerprint_tokenizer(different_pattern)


def test_checkpoint_schema_one_requires_regeneration_and_retraining():
    metadata = MTPCheckpointMetadata(
        base_model_repository="local",
        base_model_revision="rev",
        base_model_fingerprint="base",
        decoder_hidden_size=8,
        vocabulary_size=32,
        tokenizer_fingerprint="tokenizer",
        horizons=2,
        weight_dtype="float32",
        training_config={},
        label_manifest_digest="labels",
        dataset_split_digest="split",
        training_commit="commit",
        validation_metrics={},
        schema_version=1,
    )
    with pytest.raises(ValueError, match="regenerate labels and retrain"):
        validate_checkpoint_metadata(
            metadata,
            base_model_fingerprint="base",
            tokenizer_fingerprint="tokenizer",
            decoder_hidden_size=8,
            vocabulary_size=32,
        )
