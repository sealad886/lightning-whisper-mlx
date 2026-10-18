"""Synthetic native proof of the same local-model training entrypoint users run."""

from dataclasses import asdict
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import pytest
import tiktoken

from lightning_whisper_mlx.tokenizer import Tokenizer
from lightning_whisper_mlx.whisper import ModelDimensions, Whisper
from lightning_whisper_mlx.mtp import fingerprint_tokenizer, load_mtp_head
from training import train_mtp
from training.manifest import build_manifest


def tiny_assets(tmp_path, monkeypatch):
    specials = {
        "<|endoftext|>": 256,
        "<|startoftranscript|>": 257,
        "<|en|>": 258,
        "<|translate|>": 259,
        "<|transcribe|>": 260,
        "<|startoflm|>": 261,
        "<|startofprev|>": 262,
        "<|nospeech|>": 263,
        "<|notimestamps|>": 264,
        **{f"<|{i * .02:.2f}|>": 265 + i for i in range(23)},
    }
    encoding = tiktoken.Encoding(
        name="training-fixture",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens=specials,
        explicit_n_vocab=288,
    )
    tokenizer = Tokenizer(
        encoding=encoding, num_languages=1, language="en", task="transcribe"
    )
    monkeypatch.setattr(train_mtp, "get_tokenizer", lambda *args, **kwargs: tokenizer)
    import lightning_whisper_mlx.decoding as decoding

    monkeypatch.setattr(decoding, "get_tokenizer", lambda *args, **kwargs: tokenizer)
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=6,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=0,
        n_vocab=288,
        n_text_ctx=16,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims, dtype=mx.float32)
    # Keep a real decoder block/cache while making its residual update zero;
    # positional states then deterministically emit ABC and natural EOT.
    for block in model.decoder.blocks:
        block.update(
            tree_unflatten(
                [
                    (name, mx.zeros_like(value))
                    for name, value in tree_flatten(block.parameters())
                ]
            )
        )
    embeddings = np.zeros((288, 8), dtype=np.float32)
    for axis, token in enumerate((65, 66, 67, tokenizer.eot)):
        embeddings[token, axis] = 1.0
    model.decoder.token_embedding.weight = mx.array(embeddings)
    positions = np.zeros((16, 8), dtype=np.float32)
    prefix_length = len(tokenizer.sot_sequence_including_notimestamps)
    for axis in range(4):
        positions[prefix_length - 1 + axis, axis] = 10.0
    model.decoder.positional_embedding = mx.array(positions)
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(asdict(dims)))
    mx.savez(str(target / "weights.npz"), **dict(tree_flatten(model.parameters())))
    cache = tmp_path / "cache"
    shard = cache / "speech/shard_000"
    shard.mkdir(parents=True)
    rng = np.random.default_rng(42)
    records = []
    for index in range(10):
        name = f"audio_{index}.npy"
        np.save(shard / name, rng.normal(0, 0.01, 1600).astype(np.float32))
        records.append(
            {"audio_path": name, "speaker_id": f"speaker-{index}", "sample_rate": 16000}
        )
    (shard / "__paths__.json").write_text(json.dumps(records))
    return cache, target, tokenizer


def test_local_cache_to_trained_checkpoint_and_reload_is_frozen_and_held_out(
    tmp_path, monkeypatch
):
    cache, target, tokenizer = tiny_assets(tmp_path, monkeypatch)
    original_loader = train_mtp.load_model
    loaded_targets = []

    def record_load(*args, **kwargs):
        model = original_loader(*args, **kwargs)
        loaded_targets.append(
            (
                model,
                {
                    name: mx.array(value)
                    for name, value in tree_flatten(model.parameters())
                },
            )
        )
        mx.eval(loaded_targets[-1][1])
        return model

    monkeypatch.setattr(train_mtp, "load_model", record_load)
    run = tmp_path / "run"
    checkpoint = tmp_path / "head"
    arguments = [
        "--cache-root",
        str(cache),
        "--model",
        str(target),
        "--run-dir",
        str(run),
        "--checkpoint",
        str(checkpoint),
        "--epochs",
        "2",
        "--batch-size",
        "3",
        "--seed",
        "7",
        "--max-tokens",
        "8",
    ]
    assert train_mtp.main(arguments) == 0
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["validation_metrics"]["validation_examples"] > 0
    assert np.isfinite(metadata["validation_metrics"]["loss"])
    config = metadata["training_config"]
    assert len(config["epoch_history"]) == 2
    assert config["epoch_history"][0]["training_examples"] > 0
    assert set(config["selected_audio_digests"]["train"]).isdisjoint(
        config["selected_audio_digests"]["validation"]
    )
    assert config["source_digests"]["training/train_mtp.py"]
    model, before = loaded_targets[0]
    assert not tree_flatten(model.trainable_parameters())
    assert all(
        mx.array_equal(before[name], value).item()
        for name, value in tree_flatten(model.parameters())
    )
    labels = [
        json.loads(line)
        for line in (run / "labels/labels.jsonl").read_text().splitlines()
    ]
    expected_prefix = list(tokenizer.sot_sequence_including_notimestamps)
    assert labels and all(
        row["tokens"] == expected_prefix + [65, 66, 67, tokenizer.eot] for row in labels
    )
    loaded, _ = load_mtp_head(
        str(checkpoint),
        decoder_hidden_size=8,
        vocabulary_size=288,
        base_model_fingerprint=model._checkpoint_fingerprint,
        tokenizer_fingerprint=fingerprint_tokenizer(tokenizer),
    )
    assert loaded.horizons == 2
    # Label resume is useful; training starts deterministically from a new head,
    # rather than pretending this inference checkpoint contains optimizer state.
    second = tmp_path / "head-second"
    arguments[arguments.index(str(checkpoint))] = str(second)
    assert train_mtp.main(arguments) == 0
    repeated = json.loads((second / "metadata.json").read_text())
    assert repeated["validation_metrics"] == metadata["validation_metrics"]
    assert (second / "weights.npz").read_bytes() == (
        checkpoint / "weights.npz"
    ).read_bytes()


def test_quantized_tied_projection_trains_only_head():
    embedding = nn.Embedding(32, 64).to_quantized(group_size=32, bits=4)
    embedding.freeze()
    before = {
        name: mx.array(value) for name, value in tree_flatten(embedding.parameters())
    }
    trainer = train_mtp.MTPTrainer(
        64, 32, 2, 0.01, output_projection=embedding.as_linear
    )
    head_before = {
        name: mx.array(value) for name, value in tree_flatten(trainer.head.parameters())
    }
    mx.eval(before, head_before)
    loss = trainer.train_step(
        mx.random.normal((3, 64)), mx.array([[1, 2], [3, 4], [5, 6]])
    )
    assert np.isfinite(loss)
    assert any(
        not mx.array_equal(head_before[name], value).item()
        for name, value in tree_flatten(trainer.head.parameters())
    )
    assert all(
        mx.array_equal(before[name], value).item()
        for name, value in tree_flatten(embedding.parameters())
    )


@pytest.mark.parametrize("damage", ["empty", "nonfinite", "wrong_rate", "too_long"])
def test_manifest_rejects_invalid_cache_audio_without_overwriting_output(
    tmp_path, damage
):
    shard = tmp_path / "cache/speech/shard_000"
    shard.mkdir(parents=True)
    audio = np.ones(160, dtype=np.float32)
    if damage == "empty":
        audio = np.array([], dtype=np.float32)
    if damage == "nonfinite":
        audio[0] = np.nan
    if damage == "too_long":
        audio = np.ones(480001, dtype=np.float32)
    np.save(shard / "audio.npy", audio)
    (shard / "__paths__.json").write_text(
        json.dumps(
            [
                {
                    "audio_path": "audio.npy",
                    "sample_rate": 8000 if damage == "wrong_rate" else 16000,
                }
            ]
        )
    )
    output = tmp_path / "manifest.jsonl"
    output.write_text("previous manifest")
    with pytest.raises(ValueError, match="invalid source audio"):
        build_manifest(tmp_path / "cache", output)
    assert output.read_text() == "previous manifest"


def test_pipeline_refuses_remote_model_and_source_output_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_mtp,
        "load_model",
        lambda *args, **kwargs: pytest.fail("must not download"),
    )
    with pytest.raises(ValueError, match="downloads are disabled"):
        train_mtp.run_training(
            cache_root=tmp_path / "cache",
            model_path=Path("org/remote"),
            run_dir=tmp_path / "run",
            checkpoint=tmp_path / "head",
        )
    target = tmp_path / "target"
    target.mkdir()
    for name in ("config.json", "weights.npz"):
        (target / name).touch()
    with pytest.raises(ValueError, match="outside the read-only source cache"):
        train_mtp.run_training(
            cache_root=tmp_path / "cache",
            model_path=target,
            run_dir=tmp_path / "cache/run",
            checkpoint=tmp_path / "head",
        )


def test_teacher_forced_features_match_complete_prefix_hidden_states(
    tmp_path, monkeypatch
):
    from training.dataset import iter_teacher_examples
    from training.manifest import read_manifest
    from training.pseudo_label import generate_pseudo_labels
    from dataclasses import asdict
    from lightning_whisper_mlx.decoding import DecodingOptions

    cache, target, tokenizer = tiny_assets(tmp_path, monkeypatch)
    model = train_mtp.load_model(str(target), dtype=mx.float32)
    model.freeze()
    manifest_path = tmp_path / "manifest.jsonl"
    build_manifest(cache, manifest_path, seed=1, max_samples=1920)
    records = read_manifest(manifest_path)
    options = DecodingOptions(
        language="en", without_timestamps=True, fp16=False, sample_len=8
    )
    identity = {
        "schema_version": 2,
        "teacher_fingerprint": model._checkpoint_fingerprint,
        "tokenizer_fingerprint": fingerprint_tokenizer(tokenizer),
        "language": "en",
        "task": "transcribe",
        "decoder_config": asdict(options),
        "dataset_split_digest": train_mtp._digest(manifest_path),
        "vocabulary_size": 288,
        "prefix": list(tokenizer.sot_sequence_including_notimestamps),
        "eot": tokenizer.eot,
    }
    labels_dir = tmp_path / "labels"
    generate_pseudo_labels(
        records, train_mtp._teacher(model, options), labels_dir, run_identity=identity
    )
    examples = list(iter_teacher_examples(records, labels_dir / "labels.jsonl", 2))
    hidden, targets = next(
        train_mtp._feature_batches(
            model, records, labels_dir / "labels.jsonl", 2, 2, "train"
        )
    )
    for index, example in enumerate(examples[:2]):
        audio = model.embed_audio(train_mtp._audio_mel(example, model)[None])
        _, prefix_hidden = model.logits_and_hidden(mx.array([example["prefix"]]), audio)
        assert mx.allclose(hidden[index], prefix_hidden[0, -1], atol=1e-6).item()
        assert targets[index].tolist() == example["targets"]
        prefix_length = len(example["prefix"])
        assert (
            example["targets"]
            == example["teacher_tokens"][prefix_length + 1 : prefix_length + 3]
        )


def test_validation_requires_usable_held_out_examples():
    trainer = train_mtp.MTPTrainer(
        4, 8, 2, 0.01, output_projection=nn.Embedding(8, 4).as_linear
    )
    with pytest.raises(ValueError, match="validation split has no usable"):
        train_mtp._validation(trainer, iter(()))
