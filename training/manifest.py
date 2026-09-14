"""Streaming speech-cache manifest generation and grouped splitting."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_records(path: Path) -> Iterable[Dict[str, object]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("paths", []))
    for record in payload:
        if isinstance(record, str):
            yield {"audio_path": record}
        else:
            yield dict(record)


def _record_from_metadata(shard: Path, metadata: Dict[str, object]) -> Dict[str, object]:
    shard = shard.resolve()
    audio_path = (shard / Path(str(metadata["audio_path"]))).resolve()
    try:
        relative = audio_path.relative_to(shard)
    except ValueError as error:
        raise ValueError("metadata audio_path escapes its source shard") from error
    record_id = hashlib.sha256(
        f"{shard.name}:{relative.as_posix()}".encode()
    ).hexdigest()[:24]
    status = "valid"
    try:
        array = np.load(audio_path, mmap_mode="r", allow_pickle=False)
        shape = list(array.shape)
        dtype = str(array.dtype)
        sample_count = int(array.size)
        if array.ndim != 1 or array.dtype.kind != "f":
            status = "invalid_shape_or_dtype"
    except (OSError, ValueError):
        shape = []
        dtype = "unknown"
        sample_count = 0
        status = "unreadable"

    sample_rate = int(metadata.get("sample_rate", 16000))
    if sample_rate <= 0:
        status = "unknown_sample_rate"
    audio_digest = _digest(audio_path) if status != "unreadable" else ""
    return {
        "record_id": record_id,
        "relative_audio_path": relative.as_posix(),
        "source_path": str(shard),
        "source_dataset": metadata.get("source_dataset", "unknown"),
        "speaker_id": metadata.get("speaker_id", "unknown"),
        "chapter_or_session": metadata.get("chapter_or_session"),
        "sample_count": sample_count,
        "sample_rate": sample_rate,
        "duration": sample_count / sample_rate if sample_rate else 0.0,
        "npy_dtype": dtype,
        "npy_shape": shape,
        "audio_digest": audio_digest,
        "validation_status": status,
        "cache_generation": metadata.get("cache_generation"),
    }


def split_records(records: List[Dict[str, object]], seed: int = 0) -> List[Dict[str, object]]:
    groups: Dict[str, set] = {}
    for index, record in enumerate(records):
        speaker = record.get("speaker_id")
        digest = str(record.get("audio_digest", record.get("record_id", index)))
        if speaker is not None and str(speaker).strip().lower() not in ("", "unknown"):
            groups.setdefault(f"speaker:{speaker}", set()).add(index)
        groups.setdefault(f"digest:{digest}", set()).add(index)

    components: List[set] = []
    remaining = set(range(len(records)))
    while remaining:
        component = {remaining.pop()}
        changed = True
        while changed:
            changed = False
            for indices in groups.values():
                if component.intersection(indices) and not indices.issubset(component):
                    component.update(indices)
                    remaining.difference_update(indices)
                    changed = True
        components.append(component)

    components.sort(key=lambda component: min(component))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(components)).tolist()
    totals = [0, 0, 0]
    targets = [0.8 * len(records), 0.1 * len(records), 0.1 * len(records)]
    names = ["train", "validation", "test"]
    assignments = {}
    for index in order:
        split_index = min(range(3), key=lambda candidate: totals[candidate] - targets[candidate])
        for record_index in components[index]:
            assignments[record_index] = names[split_index]
        totals[split_index] += len(components[index])

    return [dict(record, split=assignments[index]) for index, record in enumerate(records)]


def build_manifest(cache_root: Path, output: Path, seed: int = 0) -> int:
    records = []
    for metadata_path in sorted((cache_root / "speech").glob("shard_*/__paths__.json")):
        shard = metadata_path.parent
        for metadata in _metadata_records(metadata_path):
            record = _record_from_metadata(shard, metadata)
            if record["validation_status"] == "valid":
                records.append(record)
    records = split_records(records, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(output)
    return len(records)


def read_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
