from training.pseudo_label import generate_pseudo_labels


def test_pseudo_labeling_is_resumable_and_records_failures(tmp_path):
    records = [
        {"audio_digest": "a", "relative_audio_path": "a.wav"},
        {"audio_digest": "b", "relative_audio_path": "b.wav"},
    ]

    def teacher(record):
        if record["audio_digest"] == "b":
            raise RuntimeError("decode failed")
        return {"tokens": [1, 2], "text": "ok", "avg_logprob": -0.1}

    first = generate_pseudo_labels(records, teacher, tmp_path)
    second = generate_pseudo_labels(records, teacher, tmp_path)

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
        return {"tokens": [1, 2]}

    with pytest.raises(KeyboardInterrupt):
        generate_pseudo_labels(records, teacher, tmp_path, flush_every=3)
    failures = (tmp_path / "failures.jsonl").read_bytes()
    assert calls == ["0", "1", "2", "3", "4"]
    resumed = []
    counts = generate_pseudo_labels(
        records,
        lambda record: resumed.append(record["audio_digest"]) or {"tokens": [1, 2]},
        tmp_path,
        flush_every=3,
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
        return {"tokens": object() if record["audio_digest"] == "b" else [1, 2]}

    counts = generate_pseudo_labels(records, teacher, tmp_path, flush_every=2)

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
