import json
from pathlib import Path

import pytest

import numpy as np

from training.manifest import build_manifest, read_manifest, split_records


def test_manifest_streams_metadata_and_groups_splits(tmp_path):
    shard = tmp_path / "speech" / "shard_000"
    shard.mkdir(parents=True)
    np.save(shard / "audio_0.npy", np.zeros(160, dtype=np.float32))
    np.save(shard / "audio_1.npy", np.ones(160, dtype=np.float32))
    (shard / "__paths__.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "audio_path": "audio_0.npy",
                        "speaker_id": "speaker-a",
                        "source_dataset": "vctk",
                    },
                    {
                        "audio_path": "audio_1.npy",
                        "speaker_id": "speaker-a",
                        "source_dataset": "vctk",
                    },
                ]
            }
        )
    )

    output = tmp_path / "manifest.jsonl"
    count = build_manifest(tmp_path, output, seed=7)
    records = read_manifest(output)

    assert count == 2
    assert len(records) == 2
    assert all(record["validation_status"] == "valid" for record in records)
    assert len({record["split"] for record in records}) == 1


def test_split_records_never_separates_duplicate_digests():
    records = [
        {"record_id": "a", "speaker_id": "one", "audio_digest": "same"},
        {"record_id": "b", "speaker_id": "two", "audio_digest": "same"},
        {"record_id": "c", "speaker_id": "three", "audio_digest": "other"},
    ]

    split = split_records(records, seed=1)
    by_digest = {}
    for record in split:
        by_digest.setdefault(record["audio_digest"], set()).add(record["split"])

    assert all(len(splits) == 1 for splits in by_digest.values())


def test_split_records_balances_independent_records_reproducibly():
    records = [
        {"record_id": str(i), "speaker_id": str(i), "audio_digest": str(i)}
        for i in range(100)
    ]

    split = split_records(records, seed=7)

    assert {
        name: sum(record["split"] == name for record in split)
        for name in ("train", "validation", "test")
    } == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    assert split == split_records(records, seed=7)
    assert all("split" not in record for record in records)


def test_split_records_keeps_transitive_speaker_and_digest_groups_together():
    records = [
        {"record_id": "a", "speaker_id": "one", "audio_digest": "first"},
        {"record_id": "b", "speaker_id": "one", "audio_digest": "shared"},
        {"record_id": "c", "speaker_id": "two", "audio_digest": "shared"},
    ]

    split = split_records(records, seed=7)

    assert len({record["split"] for record in split}) == 1


def test_unknown_speakers_do_not_form_one_split_group():
    records = [
        {"audio_digest": str(i), "speaker_id": [None, "", "unknown", " "][i % 4]}
        for i in range(100)
    ]
    split = split_records(records, seed=3)
    assert [
        sum(record["split"] == name for record in split)
        for name in ("train", "validation", "test")
    ] == [80, 10, 10]


def test_unknown_speakers_still_group_duplicate_digests():
    records = [
        {"audio_digest": "shared"},
        {"audio_digest": "shared", "speaker_id": "known"},
        {"audio_digest": "other", "speaker_id": "known"},
    ]
    assert len({record["split"] for record in split_records(records)}) == 1


def test_manifest_source_references_resolve_independently_of_working_directory(
    tmp_path, monkeypatch
):
    shard = tmp_path / "cache" / "speech" / "shard_000"
    nested = shard / "nested"
    nested.mkdir(parents=True)
    audio = nested / "audio.npy"
    np.save(audio, np.ones(160, dtype=np.float32))
    (shard / "__paths__.json").write_text(
        json.dumps(
            [
                {"audio_path": "nested/audio.npy"},
                {"audio_path": str(audio)},
            ]
        )
    )
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    monkeypatch.chdir(invocation_dir)
    output = tmp_path / "manifest.jsonl"

    assert build_manifest(Path("../cache"), output) == 2
    monkeypatch.chdir(tmp_path)
    records = read_manifest(output)
    for record in records:
        assert record["source_path"] == str(shard.resolve())
        assert record["relative_audio_path"] == "nested/audio.npy"
        reference = Path(record["source_path"]) / record["relative_audio_path"]
        assert reference == audio
        assert np.array_equal(np.load(reference), np.ones(160, dtype=np.float32))
    assert records[0]["record_id"] == records[1]["record_id"]


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_manifest_rejects_escaping_audio_before_reading(tmp_path, monkeypatch, escape):
    shard = tmp_path / "speech" / "shard_000"
    shard.mkdir(parents=True)
    outside = tmp_path / "outside.npy"
    if escape == "parent":
        audio_path = "../../outside.npy"
    elif escape == "absolute":
        audio_path = str(outside)
    else:
        (shard / "linked.npy").symlink_to(outside)
        audio_path = "linked.npy"
    (shard / "__paths__.json").write_text(json.dumps([{"audio_path": audio_path}]))
    output = tmp_path / "manifest.jsonl"
    output.write_text("previous manifest")

    def unexpected_audio_read(*args, **kwargs):
        pytest.fail("escaping source audio must not be read")

    monkeypatch.setattr(np, "load", unexpected_audio_read)
    with pytest.raises(ValueError, match="escapes"):
        build_manifest(tmp_path, output)
    assert output.read_text() == "previous manifest"
