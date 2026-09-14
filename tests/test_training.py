import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from lightning_whisper_mlx.mtp import load_mtp_head

from training.train_mtp import MTPTrainer


def test_mtp_trainer_updates_only_head_and_returns_loss():
    trainer = MTPTrainer(
        hidden_size=4, vocabulary_size=8, horizons=2, learning_rate=0.01
    )
    hidden = mx.random.normal((1, 2, 4))
    embedding = mx.random.normal((8, 4))
    targets = mx.array([[[1, 2], [3, 4]]])
    before = {
        name: mx.array(value) for name, value in tree_flatten(trainer.head.parameters())
    }
    embedding_before = mx.array(embedding)
    mx.eval(before, embedding_before)

    loss = trainer.train_step(hidden, embedding, targets)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert any(
        not mx.array_equal(before[name], value).item()
        for name, value in tree_flatten(trainer.head.parameters())
    )
    assert mx.array_equal(embedding, embedding_before).item()


def test_mtp_trainer_writes_runtime_loadable_checkpoint(tmp_path, monkeypatch):
    trainer = MTPTrainer(
        hidden_size=4, vocabulary_size=8, horizons=2, learning_rate=0.01
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

    assert (checkpoint / "metadata.json").exists()
    assert (checkpoint / "weights.npz").exists()

    loaded, _ = load_mtp_head(
        str(checkpoint),
        decoder_hidden_size=4,
        vocabulary_size=8,
        base_model_fingerprint="base",
        tokenizer_fingerprint="tokenizer",
    )
    hidden = mx.ones((1, 1, 4))
    embedding = mx.ones((8, 4))
    def output_projection(future_hidden):
        return future_hidden @ embedding.T

    assert mx.array_equal(
        loaded(hidden, output_projection), trainer.head(hidden, output_projection)
    ).item()

    original = {file.name: file.read_bytes() for file in checkpoint.iterdir()}
    arguments = dict(
        base_model_repository="org/model",
        base_model_revision="rev",
        base_model_fingerprint="base",
        tokenizer_fingerprint="tokenizer",
        label_manifest_digest="labels",
        dataset_split_digest="split",
        training_commit="commit",
    )
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(checkpoint, **arguments)

    def fail_metadata(self, *args, **kwargs):
        raise OSError("metadata write failed")

    monkeypatch.setattr(type(checkpoint), "write_text", fail_metadata)
    with pytest.raises(OSError, match="metadata write failed"):
        trainer.save_checkpoint(tmp_path / "next", **arguments)
    assert not (tmp_path / "next").exists()
    assert {file.name: file.read_bytes() for file in checkpoint.iterdir()} == original
