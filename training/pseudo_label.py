"""Resumable teacher labels bound to one explicit, immutable labeling run."""

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable
from uuid import uuid4

IDENTITY_FIELDS = (
    "schema_version",
    "teacher_fingerprint",
    "tokenizer_fingerprint",
    "language",
    "task",
    "decoder_config",
    "dataset_split_digest",
    "vocabulary_size",
    "prefix",
    "eot",
)


def identity_digest(identity):
    validate_run_identity(identity)
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_run_identity(identity):
    missing = set(IDENTITY_FIELDS) - identity.keys()
    if missing:
        raise ValueError(f"label run identity missing fields: {sorted(missing)}")
    if identity["schema_version"] != 2:
        raise ValueError("label run identity requires schema_version=2")
    for key in ("teacher_fingerprint", "tokenizer_fingerprint", "dataset_split_digest"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"label run identity requires nonempty {key}")
    if not isinstance(identity["decoder_config"], dict):
        raise ValueError("label run identity decoder_config must be a dictionary")
    if identity["task"] not in ("transcribe", "translate") or not identity["language"]:
        raise ValueError("label run identity requires explicit language and task")
    vocabulary = identity["vocabulary_size"]
    if type(vocabulary) is not int or vocabulary < 1:
        raise ValueError("label run identity vocabulary_size must be positive")
    prefix = identity["prefix"]
    if not isinstance(prefix, list) or not prefix:
        raise ValueError("label run identity requires canonical prefix token IDs")
    if any(type(token) is not int or not 0 <= token < vocabulary for token in prefix):
        raise ValueError("label run identity prefix contains invalid token IDs")
    eot = identity["eot"]
    if type(eot) is not int or not 0 <= eot < vocabulary or eot in prefix:
        raise ValueError("label run identity contains invalid EOT")
    json.dumps(identity, sort_keys=True, allow_nan=False)


def validate_label(label, identity):
    tokens = label.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("teacher label requires a nonempty tokens list")
    if any(
        type(token) is not int or not 0 <= token < identity["vocabulary_size"]
        for token in tokens
    ):
        raise ValueError("teacher label token IDs must be integers within vocabulary")
    prefix = identity["prefix"]
    if label.get("prefix") != prefix or tokens[: len(prefix)] != prefix:
        raise ValueError("teacher label must include the canonical prefix")
    if len(tokens) <= len(prefix) or tokens[-1] != identity["eot"]:
        raise ValueError("teacher label must terminate with EOT after the prefix")
    if identity["eot"] in tokens[:-1]:
        raise ValueError("teacher label contains a premature EOT")
    if label.get("failure_status"):
        raise ValueError("teacher label reports a failure status")
    json.dumps(label, sort_keys=True, allow_nan=False)


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid ledger {path}:{number}; "
                        "restore a backup before resuming"
                    ) from error
    return records


def _append_jsonl(path: Path, records):
    with path.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _write_identity(path, identity):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, sort_keys=True, indent=2))
    temporary.replace(path)


def migrate_legacy_labels(output_dir: Path, *, explicit_identity: Dict[str, object]):
    """Back up and rewrite ledgers only under caller-attested legacy provenance.

    Never infer teacher provenance or invent missing token prefix/EOT. Incomplete
    legacy labels must be regenerated into a new run directory instead.
    """
    digest = identity_digest(explicit_identity)
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        raise ValueError("label run already has identity; migration is once-off")
    paths = [output_dir / "labels.jsonl", output_dir / "failures.jsonl"]
    records = [_read_jsonl(path) for path in paths]
    if not any(path.exists() for path in paths):
        raise ValueError("no legacy ledgers to migrate")
    for label in records[0]:
        validate_label(label, explicit_identity)
    backup = output_dir / f"legacy-backup-{uuid4().hex}"
    backup.mkdir()
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup / path.name)
    with TemporaryDirectory(prefix=".label-migration-", dir=output_dir) as directory:
        for path, rows in zip(paths, records):
            rewritten = Path(directory) / path.name
            _append_jsonl(
                rewritten, [dict(row, label_run_digest=digest) for row in rows]
            )
            rewritten.replace(path)
    _write_identity(identity_path, explicit_identity)
    return backup


def generate_pseudo_labels(
    records: Iterable[Dict[str, object]],
    teacher: Callable[[Dict[str, object]], Dict[str, object]],
    output_dir: Path,
    *,
    run_identity: Dict[str, object],
    flush_every: int = 32,
):
    if flush_every < 1:
        raise ValueError("flush_every must be at least 1")
    digest = identity_digest(run_identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    failures_path = output_dir / "failures.jsonl"
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != run_identity:
            raise ValueError("label run identity mismatch; use a new output directory")
    elif labels_path.exists() or failures_path.exists():
        raise ValueError(
            "legacy ledgers lack identity; regenerate in a new directory, or migrate "
            "with explicitly attested teacher/tokenizer/decoder identity"
        )
    labels = _read_jsonl(labels_path)
    failures = _read_jsonl(failures_path)
    for label in labels:
        validate_label(label, run_identity)
    if any(row.get("label_run_digest") != digest for row in labels + failures):
        raise ValueError("ledger label run identity mismatch")
    if not identity_path.exists():
        _write_identity(identity_path, run_identity)
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
            audio_digest = record["audio_digest"]
            if audio_digest in labeled_digests:
                counts["skipped"] += 1
                continue
            if audio_digest in failed_digests:
                counts["failed"] += 1
                continue
            try:
                label = teacher(record)
                validate_label(label, run_identity)
                pending_label = {
                    **label,
                    **record,
                    "label_run_digest": digest,
                }
                pending_labels.append(pending_label)
                labeled_digests.add(audio_digest)
                counts["labeled"] += 1
            except Exception as error:
                pending_failures.append(
                    {
                        "audio_digest": audio_digest,
                        "label_run_digest": digest,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                failed_digests.add(audio_digest)
                counts["failed"] += 1
            if len(pending_labels) + len(pending_failures) >= flush_every:
                flush()
    finally:
        flush()
    return counts
