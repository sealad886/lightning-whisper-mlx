"""Teacher-forced windows over explicitly selected, disjoint manifest splits."""

from pathlib import Path
from typing import Dict, Iterable

from .pseudo_label import _read_jsonl


def iter_teacher_examples(
    manifest: Iterable[Dict[str, object]],
    labels_path: Path,
    horizons: int,
    *,
    split: str = "train",
):
    if horizons < 1:
        raise ValueError("horizons must be at least 1")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    labels = {record["audio_digest"]: record for record in _read_jsonl(labels_path)}
    for record in manifest:
        if record.get("split") != split:
            continue
        label = labels.get(record.get("audio_digest"))
        if not label or label.get("failure_status"):
            continue
        tokens = label["tokens"]
        prefix = label["prefix"]
        if not prefix or tokens[: len(prefix)] != prefix:
            raise ValueError("teacher labels require their canonical prefix")
        # Prefix must be complete before the first generated token is predicted.
        # Target supplies immediate next token; horizon zero predicts the next one.
        for prefix_length in range(len(prefix), len(tokens) - horizons):
            yield {
                **record,
                "teacher_tokens": tokens,
                "prefix": tokens[:prefix_length],
                "targets": tokens[prefix_length + 1 : prefix_length + horizons + 1],
            }
