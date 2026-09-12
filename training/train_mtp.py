"""Small resumable MLX-native MTP trainer."""

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn

from lightning_whisper_mlx.mtp import MTPHead


class MTPTrainer:
    def __init__(self, hidden_size: int, vocabulary_size: int, horizons: int, learning_rate: float):
        self.head = MTPHead(hidden_size, horizons)
        self.optimizer = optim.Adam(learning_rate=learning_rate)
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

    def save_checkpoint(self, path: Path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        weights = self.head.parameters()
        mx.savez(str(temporary), **_flatten_weights(weights))
        temporary.replace(path)


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
