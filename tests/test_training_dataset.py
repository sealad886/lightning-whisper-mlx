import json

from training.dataset import iter_teacher_examples


def test_teacher_examples_align_with_future_proposal_horizons(tmp_path):
    labels = tmp_path / "labels.jsonl"
    tokens = [1, 2, 3, 4, 5]
    labels.write_text(json.dumps({"audio_digest": "a", "tokens": tokens}) + "\n")
    manifest = [{"audio_digest": "a", "split": "train"}]

    examples = list(iter_teacher_examples(manifest, labels, horizons=2))

    assert len(examples) == 2
    assert [example["prefix"] for example in examples] == [[1], [1, 2]]
    for example in examples:
        prefix_length = len(example["prefix"])
        # Runtime prepends the target decoder's next token to the head outputs.
        proposal = [tokens[prefix_length], *example["targets"]]
        assert proposal == tokens[prefix_length : prefix_length + 3]


def test_teacher_examples_skip_incomplete_horizons_and_non_training_labels(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"audio_digest": "short", "tokens": [1, 2, 3]},
                {"audio_digest": "held_out", "tokens": [1, 2, 3, 4]},
                {
                    "audio_digest": "failed",
                    "tokens": [1, 2, 3, 4],
                    "failure_status": "error",
                },
            ]
        )
    )
    manifest = [
        {"audio_digest": "short", "split": "train"},
        {"audio_digest": "held_out", "split": "validation"},
        {"audio_digest": "failed", "split": "train"},
        {"audio_digest": "missing", "split": "train"},
    ]

    assert list(iter_teacher_examples(manifest, labels, horizons=2)) == []
