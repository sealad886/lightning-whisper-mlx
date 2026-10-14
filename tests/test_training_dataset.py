import json

import pytest

from training.dataset import iter_teacher_examples


def test_teacher_examples_align_with_future_proposal_horizons(tmp_path):
    labels = tmp_path / "labels.jsonl"
    tokens = [1, 2, 3, 4, 5]
    labels.write_text(
        json.dumps({"audio_digest": "a", "tokens": tokens, "prefix": [1]}) + "\n"
    )
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
                {"audio_digest": "short", "tokens": [1, 2, 3], "prefix": [1]},
                {"audio_digest": "held_out", "tokens": [1, 2, 3, 4], "prefix": [1]},
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


@pytest.mark.parametrize("horizons", [0, -1, -10])
def test_teacher_examples_reject_nonpositive_horizons_before_reading_labels(
    tmp_path, horizons
):
    with pytest.raises(ValueError, match="horizons must be at least 1"):
        list(iter_teacher_examples([], tmp_path / "missing.jsonl", horizons=horizons))


def test_validation_windows_are_disjoint_and_begin_at_complete_prefix(tmp_path):
    labels = tmp_path / "labels.jsonl"
    rows = [
        {"audio_digest": digest, "prefix": [1, 2], "tokens": [1, 2, 3, 4, 5, 6]}
        for digest in ("training", "heldout")
    ]
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = [
        {"audio_digest": "training", "split": "train"},
        {"audio_digest": "heldout", "split": "validation"},
    ]
    train = list(iter_teacher_examples(manifest, labels, 2))
    validation = list(iter_teacher_examples(manifest, labels, 2, split="validation"))
    assert {row["audio_digest"] for row in train} == {"training"}
    assert {row["audio_digest"] for row in validation} == {"heldout"}
    assert validation[0]["prefix"] == [1, 2]
    assert validation[0]["targets"] == [4, 5]
