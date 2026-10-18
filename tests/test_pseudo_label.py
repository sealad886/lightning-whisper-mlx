import pytest

from training.pseudo_label import generate_pseudo_labels

IDENTITY = {
    "schema_version": 2,
    "teacher_fingerprint": "teacher-a",
    "tokenizer_fingerprint": "tokenizer-a",
    "dataset_split_digest": "split-a",
    "language": "en",
    "task": "transcribe",
    "decoder_config": {"temperature": 0.0},
    "vocabulary_size": 8,
    "prefix": [1],
    "eot": 7,
}


def test_pseudo_labeling_is_resumable_and_records_failures(tmp_path):
    records = [
        {"audio_digest": "a", "relative_audio_path": "a.wav"},
        {"audio_digest": "b", "relative_audio_path": "b.wav"},
    ]

    def teacher(record):
        if record["audio_digest"] == "b":
            raise RuntimeError("decode failed")
        return {"tokens": [1, 2, 7], "prefix": [1], "text": "ok", "avg_logprob": -0.1}

    first = generate_pseudo_labels(records, teacher, tmp_path, run_identity=IDENTITY)
    second = generate_pseudo_labels(records, teacher, tmp_path, run_identity=IDENTITY)

    assert first == {"labeled": 1, "failed": 1, "skipped": 0}
    assert second == {"labeled": 0, "failed": 1, "skipped": 1}
    assert (tmp_path / "failures.jsonl").read_text().count("\n") == 1


def test_interrupted_labeling_flushes_progress_and_resumes_without_repeats(tmp_path):
    import json
    import pytest

    records = [{"audio_digest": str(i)} for i in reversed(range(6))]
    calls = []

    def teacher(record):
        digest = record["audio_digest"]
        calls.append(digest)
        if digest == "4":
            # Earlier batch must already be durable before interruption.
            assert len((tmp_path / "labels.jsonl").read_text().splitlines()) == 2
            raise KeyboardInterrupt
        if digest == "1":
            raise ValueError("invalid audio")
        return {"tokens": [1, 2, 7], "prefix": [1]}

    with pytest.raises(KeyboardInterrupt):
        generate_pseudo_labels(
            records, teacher, tmp_path, flush_every=3, run_identity=IDENTITY
        )
    failures = (tmp_path / "failures.jsonl").read_bytes()
    assert calls == ["0", "1", "2", "3", "4"]
    resumed = []
    counts = generate_pseudo_labels(
        records,
        lambda record: resumed.append(record["audio_digest"])
        or {"tokens": [1, 2, 7], "prefix": [1]},
        tmp_path,
        flush_every=3,
        run_identity=IDENTITY,
    )
    assert resumed == ["4", "5"]
    assert counts == {"labeled": 2, "failed": 1, "skipped": 3}
    assert (tmp_path / "failures.jsonl").read_bytes() == failures
    labels = [
        json.loads(line)
        for line in (tmp_path / "labels.jsonl").read_text().splitlines()
    ]
    assert [record["audio_digest"] for record in labels] == ["0", "2", "3", "4", "5"]


def test_nonserializable_teacher_output_is_a_per_record_failure(tmp_path):
    import json

    records = [{"audio_digest": digest} for digest in ("a", "b", "c")]

    def teacher(record):
        return {
            "tokens": object() if record["audio_digest"] == "b" else [1, 2, 7],
            "prefix": [1],
        }

    counts = generate_pseudo_labels(
        records, teacher, tmp_path, flush_every=2, run_identity=IDENTITY
    )

    assert counts == {"labeled": 2, "failed": 1, "skipped": 0}
    labels = [
        json.loads(line)
        for line in (tmp_path / "labels.jsonl").read_text().splitlines()
    ]
    assert [record["audio_digest"] for record in labels] == ["a", "c"]
    failures = [
        json.loads(line)
        for line in (tmp_path / "failures.jsonl").read_text().splitlines()
    ]
    assert [record["audio_digest"] for record in failures] == ["b"]


def test_changed_teacher_identity_fails_before_resume_or_append(tmp_path):
    import pytest

    records = [{"audio_digest": "a"}]

    def teacher(record):
        return {"tokens": [1, 2, 7], "prefix": [1]}

    generate_pseudo_labels(records, teacher, tmp_path, run_identity=IDENTITY)
    original = (tmp_path / "labels.jsonl").read_bytes()
    with pytest.raises(ValueError, match="identity mismatch"):
        generate_pseudo_labels(
            records,
            lambda record: pytest.fail("mismatched teacher must not run"),
            tmp_path,
            run_identity=dict(IDENTITY, teacher_fingerprint="teacher-b"),
        )
    assert (tmp_path / "labels.jsonl").read_bytes() == original


@pytest.mark.parametrize(
    "label",
    [
        {"text": "missing tokens"},
        {"tokens": [1, True, 7], "prefix": [1]},
        {"tokens": [1, 8, 7], "prefix": [1]},
        {"tokens": [2, 3, 7], "prefix": [1]},
        {"tokens": [1, 2], "prefix": [1]},
        {"tokens": [1, 7, 2, 7], "prefix": [1]},
    ],
)
def test_invalid_token_labels_are_failures_and_never_completed(tmp_path, label):
    counts = generate_pseudo_labels(
        [{"audio_digest": "a"}],
        lambda record: label,
        tmp_path,
        run_identity=IDENTITY,
    )
    assert counts == {"labeled": 0, "failed": 1, "skipped": 0}
    assert not (tmp_path / "labels.jsonl").read_text()


def test_legacy_migration_requires_attestation_backs_up_and_preserves_tokens(tmp_path):
    import json
    import pytest
    from training.pseudo_label import migrate_legacy_labels

    row = {"audio_digest": "a", "tokens": [1, 2, 7], "prefix": [1]}
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(row) + "\n")
    original = labels.read_bytes()
    with pytest.raises(ValueError, match="legacy ledgers lack identity"):
        generate_pseudo_labels([], lambda record: {}, tmp_path, run_identity=IDENTITY)
    backup = migrate_legacy_labels(tmp_path, explicit_identity=IDENTITY)
    assert (backup / "labels.jsonl").read_bytes() == original
    migrated = json.loads(labels.read_text())
    assert migrated["tokens"] == row["tokens"]
    assert migrated["label_run_digest"]
    assert (
        generate_pseudo_labels(
            [row],
            lambda record: pytest.fail("already labeled"),
            tmp_path,
            run_identity=IDENTITY,
        )["skipped"]
        == 1
    )
    with pytest.raises(ValueError, match="once-off"):
        migrate_legacy_labels(tmp_path, explicit_identity=IDENTITY)
