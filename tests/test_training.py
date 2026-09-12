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
