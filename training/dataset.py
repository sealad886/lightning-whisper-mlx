"""Teacher-label examples derived from a manifest without copying source audio."""

import json
from pathlib import Path
from typing import Dict, Iterable


def iter_teacher_examples(
    manifest: Iterable[Dict[str, object]], labels_path: Path, horizons: int
):
    labels = {}
    with labels_path.open() as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                labels[record["audio_digest"]] = record

    for record in manifest:
        if record.get("split") != "train":
            continue
        label = labels.get(record.get("audio_digest"))
        if not label or label.get("failure_status"):
            continue
        tokens = list(label.get("tokens", []))
        for start in range(max(0, len(tokens) - horizons)):
            target = tokens[start + 1 : start + horizons + 1]
            if len(target) == horizons:
                yield {
                    "audio_digest": record["audio_digest"],
                    "prefix": tokens[: start + 1],
                    "targets": target,
                }
