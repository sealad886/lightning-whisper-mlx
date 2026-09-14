"""Resumable, metadata-only teacher labeling pipeline."""

import json
from pathlib import Path
from typing import Callable, Dict, Iterable


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _append_jsonl(path: Path, records):
    with path.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def generate_pseudo_labels(
    records: Iterable[Dict[str, object]],
    teacher: Callable[[Dict[str, object]], Dict[str, object]],
    output_dir: Path,
    *,
    flush_every: int = 32,
):
    if flush_every < 1:
        raise ValueError("flush_every must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    failures_path = output_dir / "failures.jsonl"
    labels = _read_jsonl(labels_path)
    failures = _read_jsonl(failures_path)
    labeled_digests = {record["audio_digest"] for record in labels}
    failed_digests = {record["audio_digest"] for record in failures}
    counts = {"labeled": 0, "failed": 0, "skipped": 0}

    pending_labels = []
    pending_failures = []

    def flush():
        _append_jsonl(labels_path, pending_labels)
        pending_labels.clear()
        _append_jsonl(failures_path, pending_failures)
        pending_failures.clear()

    try:
        for record in sorted(records, key=lambda value: str(value["audio_digest"])):
            digest = record["audio_digest"]
            if digest in labeled_digests:
                counts["skipped"] += 1
                continue
            if digest in failed_digests:
                counts["failed"] += 1
                continue
            try:
                label = teacher(record)
                pending_labels.append({**record, **label, "audio_digest": digest})
                labeled_digests.add(digest)
                counts["labeled"] += 1
            except Exception as error:  # record a failed ledger entry and continue
                pending_failures.append(
                    {
                        "audio_digest": digest,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                failed_digests.add(digest)
                counts["failed"] += 1

            if len(pending_labels) + len(pending_failures) >= flush_every:
                flush()
    finally:
        flush()
    return counts
