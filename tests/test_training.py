import mlx.core as mx

from training.dataset import iter_teacher_examples
from training.train_mtp import MTPTrainer


def test_teacher_examples_create_horizon_targets(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"audio_digest":"a","tokens":[1,2,3,4]}\n')
    manifest = [{"audio_digest": "a", "split": "train", "relative_audio_path": "a.npy"}]

    examples = list(iter_teacher_examples(manifest, labels, horizons=2))

    assert examples == [
        {
            "audio_digest": "a",
            "prefix": [1],
            "targets": [2, 3],
        },
        {
            "audio_digest": "a",
            "prefix": [1, 2],
            "targets": [3, 4],
        },
    ]


def test_mtp_trainer_updates_only_head_and_returns_loss():
    trainer = MTPTrainer(hidden_size=4, vocabulary_size=8, horizons=2, learning_rate=0.01)
    hidden = mx.random.normal((1, 2, 4))
    embedding = mx.random.normal((8, 4))
    targets = mx.array([[[1, 2], [3, 4]]])

    loss = trainer.train_step(hidden, embedding, targets)

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_mtp_trainer_writes_runtime_loadable_checkpoint(tmp_path):
    trainer = MTPTrainer(hidden_size=4, vocabulary_size=8, horizons=2, learning_rate=0.01)
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
