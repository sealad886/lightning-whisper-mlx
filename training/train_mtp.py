"""Small resumable MLX-native MTP trainer."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn

from lightning_whisper_mlx.mtp import MTPHead
from lightning_whisper_mlx.mtp import MTPCheckpointMetadata


class MTPTrainer:
    def __init__(self, hidden_size: int, vocabulary_size: int, horizons: int, learning_rate: float):
        self.head = MTPHead(hidden_size, horizons)
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.horizons = horizons
        mx.eval(self.head.parameters())

        def loss_fn(hidden, embedding, targets):
            logits = self.head(hidden, embedding)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            selected = mx.take_along_axis(logprobs, targets[..., None], axis=-1)
            return -selected.mean()

        self._loss_and_grad = nn.value_and_grad(self.head, loss_fn)

    def train_step(self, hidden: mx.array, embedding: mx.array, targets: mx.array) -> float:
        loss, grads = self._loss_and_grad(hidden, embedding, targets)
        self.optimizer.update(self.head, grads)
        mx.eval(self.head.parameters(), self.optimizer.state)
        return float(loss.item())

    def save_checkpoint(
        self,
        path: Path,
        *,
        base_model_repository: str,
        base_model_revision: str,
        base_model_fingerprint: str,
        tokenizer_fingerprint: str,
        label_manifest_digest: str,
        dataset_split_digest: str,
        training_commit: str,
        validation_metrics=None,
    ):
        path.mkdir(parents=True, exist_ok=True)
        metadata = MTPCheckpointMetadata(
            base_model_repository=base_model_repository,
            base_model_revision=base_model_revision,
            base_model_fingerprint=base_model_fingerprint,
            decoder_hidden_size=self.hidden_size,
            vocabulary_size=self.vocabulary_size,
            tokenizer_fingerprint=tokenizer_fingerprint,
            horizons=self.horizons,
            weight_dtype="float32",
            training_config={"learning_rate": float(self.optimizer.learning_rate.item())},
            label_manifest_digest=label_manifest_digest,
            dataset_split_digest=dataset_split_digest,
            training_commit=training_commit,
            validation_metrics=validation_metrics or {},
        )
        temporary = path / "weights.tmp"
        mx.savez(str(temporary), **_flatten_weights(self.head.parameters()))
        (path / "weights.tmp.npz").replace(path / "weights.npz")
        metadata_tmp = path / "metadata.json.tmp"
        metadata_tmp.write_text(json.dumps(metadata.to_dict(), sort_keys=True, indent=2))
        metadata_tmp.replace(path / "metadata.json")


def _flatten_weights(tree, prefix=""):
    flat = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            flat.update(_flatten_weights(value, f"{prefix}{key}."))
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            flat.update(_flatten_weights(value, f"{prefix}{index}."))
    else:
        flat[prefix[:-1]] = tree
    return flat
