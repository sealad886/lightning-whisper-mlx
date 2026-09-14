"""MTP heads and versioned checkpoint compatibility metadata."""

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


class MTPHead(nn.Module):
    def __init__(self, hidden_size: int, horizons: int):
        super().__init__()
        if horizons < 1:
            raise ValueError("horizons must be at least 1")
        self.horizons = horizons
        self.projections = [
            nn.Linear(hidden_size, hidden_size) for _ in range(horizons)
        ]
        self.norms = [nn.LayerNorm(hidden_size) for _ in range(horizons)]

    def __call__(
        self, hidden: mx.array, output_projection: Callable[[mx.array], mx.array]
    ) -> mx.array:
        outputs = []
        for projection, norm in zip(self.projections, self.norms):
            future_hidden = hidden + projection(nn.gelu(norm(hidden)))
            outputs.append(output_projection(future_hidden))
        return mx.stack(outputs, axis=-2)


@dataclass(frozen=True)
class MTPCheckpointMetadata:
    base_model_repository: str
    base_model_revision: str
    base_model_fingerprint: str
    decoder_hidden_size: int
    vocabulary_size: int
    tokenizer_fingerprint: str
    horizons: int
    weight_dtype: str
    training_config: Dict[str, object]
    label_manifest_digest: str
    dataset_split_digest: str
    training_commit: str
    validation_metrics: Dict[str, float]
    schema_version: int = 1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, object]):
        return cls(**value)


def validate_checkpoint_metadata(
    metadata: MTPCheckpointMetadata,
    *,
    base_model_fingerprint: str,
    tokenizer_fingerprint: str,
    decoder_hidden_size: int,
    vocabulary_size: int,
):
    if metadata.schema_version != 1:
        raise ValueError("unsupported MTP checkpoint schema version")
    if metadata.base_model_fingerprint != base_model_fingerprint:
        raise ValueError("MTP checkpoint base model fingerprint is incompatible")
    if metadata.tokenizer_fingerprint != tokenizer_fingerprint:
        raise ValueError("MTP checkpoint tokenizer fingerprint is incompatible")
    if metadata.decoder_hidden_size != decoder_hidden_size:
        raise ValueError("MTP checkpoint decoder hidden size is incompatible")
    if metadata.vocabulary_size != vocabulary_size:
        raise ValueError("MTP checkpoint vocabulary size is incompatible")


def load_mtp_head(
    checkpoint: str,
    *,
    decoder_hidden_size: int,
    vocabulary_size: int,
    base_model_fingerprint: str,
    tokenizer_fingerprint: str,
):
    path = Path(checkpoint)
    with (path / "metadata.json").open() as stream:
        metadata = MTPCheckpointMetadata.from_dict(json.load(stream))
    validate_checkpoint_metadata(
        metadata,
        base_model_fingerprint=base_model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        decoder_hidden_size=decoder_hidden_size,
        vocabulary_size=vocabulary_size,
    )
    head = MTPHead(decoder_hidden_size, metadata.horizons)
    weights = tree_unflatten(list(mx.load(str(path / "weights.npz")).items()))
    head.update(weights)
    mx.eval(head.parameters())
    return head, metadata


def fingerprint_tokenizer(tokenizer) -> str:
    values = {
        "eot": tokenizer.eot,
        "sot": tokenizer.sot,
        "sot_prev": tokenizer.sot_prev,
        "timestamp_begin": tokenizer.timestamp_begin,
        "transcribe": tokenizer.transcribe,
        "translate": tokenizer.translate,
        "no_timestamps": tokenizer.no_timestamps,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
