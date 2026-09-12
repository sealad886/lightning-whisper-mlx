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
