"""Bounded local-only MTP training and immutable inference checkpoint export."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from lightning_whisper_mlx.audio import HOP_LENGTH, log_mel_spectrogram, pad_or_trim
from lightning_whisper_mlx.decoding import DecodingOptions, DecodingTask
from lightning_whisper_mlx.load_models import load_model
from lightning_whisper_mlx.mtp import (
    MTPHead,
    MTPCheckpointMetadata,
    fingerprint_tokenizer,
    load_mtp_head,
)
from lightning_whisper_mlx.tokenizer import get_tokenizer
from .dataset import iter_teacher_examples
from .manifest import _digest, build_manifest, read_manifest
from .pseudo_label import generate_pseudo_labels, migrate_legacy_labels


class MTPTrainer:
    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
        horizons: int,
        learning_rate: float,
        *,
        output_projection,
    ):
        if hidden_size < 1 or vocabulary_size < 1:
            raise ValueError("hidden_size and vocabulary_size must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not callable(output_projection):
            raise ValueError("output_projection must be the frozen target's callable")
        self.head = MTPHead(hidden_size, horizons)
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.output_projection = output_projection
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.horizons = horizons
        mx.eval(self.head.parameters())
        self._loss_and_grad = nn.value_and_grad(self.head, self.loss)

    def logits(self, hidden):
        logits = self.head(mx.stop_gradient(hidden), self.output_projection)
        if logits.shape[-1] != self.vocabulary_size:
            raise ValueError("target output projection vocabulary size mismatch")
        return logits.astype(mx.float32)

    def loss(self, hidden, targets):
        return nn.losses.cross_entropy(self.logits(hidden), targets, reduction="mean")

    def train_step(self, hidden: mx.array, targets: mx.array) -> float:
        loss, grads = self._loss_and_grad(hidden, targets)
        mx.eval(loss, grads)
        value = float(loss.item())
        if not math.isfinite(value):
            raise ValueError("nonfinite training loss; checkpoint was not exported")
        self.optimizer.update(self.head, grads)
        mx.eval(self.head.parameters(), self.optimizer.state)
        return value

    def save_checkpoint(
        self,
        path: Path,
        *,
        base_model_repository: str,
        base_model_revision: str,
        base_model_fingerprint: str,
        tokenizer_fingerprint: str,
        label_manifest_digest: str,
        dataset_split_digest: str,
        training_commit: str,
        validation_metrics=None,
        training_config=None,
    ):
        if path.exists():
            raise FileExistsError(f"checkpoint destination already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = MTPCheckpointMetadata(
            base_model_repository=base_model_repository,
            base_model_revision=base_model_revision,
            base_model_fingerprint=base_model_fingerprint,
            decoder_hidden_size=self.hidden_size,
            vocabulary_size=self.vocabulary_size,
            tokenizer_fingerprint=tokenizer_fingerprint,
            horizons=self.horizons,
            weight_dtype="float32",
            training_config={
                **(training_config or {}),
                "learning_rate": float(self.optimizer.learning_rate.item()),
            },
            label_manifest_digest=label_manifest_digest,
            dataset_split_digest=dataset_split_digest,
            training_commit=training_commit,
            validation_metrics=validation_metrics or {},
        )
        with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as directory:
            temporary = Path(directory) / "checkpoint"
            temporary.mkdir()
            mx.savez(
                str(temporary / "weights.npz"),
                **dict(tree_flatten(self.head.parameters())),
            )
            (temporary / "metadata.json").write_text(
                json.dumps(
                    metadata.to_dict(), sort_keys=True, indent=2, allow_nan=False
                )
            )
            if path.exists():
                raise FileExistsError(f"checkpoint destination already exists: {path}")
            temporary.rename(path)


def _audio_mel(record, model):
    path = Path(record["source_path"]) / record["relative_audio_path"]
    if _digest(path) != record["audio_digest"]:
        raise ValueError(f"source audio changed after manifest creation: {path}")
    audio = np.load(path, mmap_mode="r", allow_pickle=False)
    max_samples = 2 * model.dims.n_audio_ctx * HOP_LENGTH
    if (
        record["sample_rate"] != 16000
        or audio.ndim != 1
        or audio.dtype.kind != "f"
        or not 0 < audio.size <= max_samples
        or not np.isfinite(audio).all()
    ):
        raise ValueError(
            f"invalid audio {path}; require finite mono 16000 Hz "
            "within target audio context"
        )
    mel = log_mel_spectrogram(
        pad_or_trim(mx.array(audio.astype(np.float32)), max_samples),
        n_mels=model.dims.n_mels,
    )
    mx.eval(mel)
    return mel


def _teacher(model, options):
    def label(record):
        task = DecodingTask(model, options)
        result = task.run(_audio_mel(record, model)[None])[0]
        if len(result.tokens) >= task.sample_len:
            raise ValueError(
                "teacher reached max_tokens without proven EOT; "
                "use a larger bound or shorter clip"
            )
        prefix = list(task.initial_tokens)
        return {
            "prefix": prefix,
            "tokens": prefix + list(result.tokens) + [task.tokenizer.eot],
            "text": result.text,
            "avg_logprob": float(result.avg_logprob),
        }

    return label


def _feature_batches(model, manifest, labels_path, horizons, batch_size, split):
    # Process one audio record at a time; retain only bounded decoder features and
    # at most batch_size hidden rows. No full-cache feature materialization.
    hidden_rows, target_rows = [], []
    previous_digest = None
    for example in iter_teacher_examples(manifest, labels_path, horizons, split=split):
        digest = example["audio_digest"]
        if digest != previous_digest:
            audio = mx.stop_gradient(
                model.embed_audio(_audio_mel(example, model)[None])
            )
            tokens = example["teacher_tokens"][:-1]
            if len(tokens) > model.dims.n_text_ctx:
                raise ValueError("teacher sequence exceeds target decoder context")
            _, all_hidden = model.logits_and_hidden(mx.array([tokens]), audio)
            all_hidden = mx.stop_gradient(all_hidden)
            mx.eval(all_hidden)
            previous_digest = digest
        if len(example["prefix"]) >= model.dims.n_text_ctx:
            raise ValueError("teacher prefix exceeds target decoder context")
        hidden_rows.append(all_hidden[0, len(example["prefix"]) - 1])
        target_rows.append(example["targets"])
        # Materialize here so pending rows cannot retain unbounded lazy graphs.
        mx.eval(hidden_rows)
        if len(hidden_rows) == batch_size:
            yield mx.stack(hidden_rows), mx.array(target_rows)
            hidden_rows, target_rows = [], []
    if hidden_rows:
        yield mx.stack(hidden_rows), mx.array(target_rows)


def _validation(trainer, batches):
    count, loss_sum = 0, 0.0
    correct = np.zeros(trainer.horizons, dtype=np.float64)
    accepted = 0.0
    for hidden, targets in batches:
        logits = trainer.logits(hidden)
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        matches = mx.argmax(logits, axis=-1) == targets
        prefix_matches = mx.cumprod(matches.astype(mx.float32), axis=-1)
        mx.eval(loss, matches, prefix_matches)
        size = targets.shape[0]
        count += size
        loss_sum += float(loss.item()) * size
        correct += np.asarray(matches).sum(axis=0)
        accepted += float(prefix_matches.sum().item())
    if count == 0:
        raise ValueError(
            "validation split has no usable horizon windows; "
            "add held-out clips or reduce horizons"
        )
    metrics = {
        "validation_examples": count,
        "loss": loss_sum / count,
        "teacher_forced_prefix_acceptance": accepted / (count * trainer.horizons),
        **{
            f"horizon_{i + 2}_accuracy": float(value / count)
            for i, value in enumerate(correct)
        },
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(
            "nonfinite held-out validation metrics; checkpoint was not exported"
        )
    return metrics


def _source_identity():
    root = Path(__file__).resolve().parent.parent
    files = [
        Path(__file__),
        root / "training/manifest.py",
        root / "training/pseudo_label.py",
        root / "training/dataset.py",
        root / "lightning_whisper_mlx/mtp.py",
        root / "lightning_whisper_mlx/decoding.py",
        root / "lightning_whisper_mlx/whisper.py",
        root / "lightning_whisper_mlx/audio.py",
        root / "lightning_whisper_mlx/tokenizer.py",
        root / "lightning_whisper_mlx/load_models.py",
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            commit += "+dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return commit, {str(path.relative_to(root)): _digest(path) for path in files}


def run_training(
    *,
    cache_root: Path,
    model_path: Path,
    run_dir: Path,
    checkpoint: Path,
    language: str = "en",
    task: str = "transcribe",
    horizons: int = 2,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 0,
    max_train_records: int = 128,
    max_validation_records: int = 32,
    max_tokens: int = None,
    migrate_identity: Path = None,
):
    """Train from selected external cache clips; never download a model or audio."""
    for name, value in (
        ("horizons", horizons),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("max_train_records", max_train_records),
        ("max_validation_records", max_validation_records),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint destination already exists: {checkpoint}")
    model_path = model_path.resolve()
    if not all(
        (model_path / name).is_file() for name in ("config.json", "weights.npz")
    ):
        raise ValueError(
            "--model must be an existing local config.json/weights.npz directory; "
            "downloads are disabled"
        )
    cache_root = cache_root.resolve()
    run_dir = run_dir.resolve()
    checkpoint = checkpoint.resolve()
    if (
        run_dir == cache_root
        or cache_root in run_dir.parents
        or checkpoint == cache_root
        or cache_root in checkpoint.parents
    ):
        raise ValueError(
            "run and checkpoint outputs must be outside the read-only source cache"
        )
    commit, sources = _source_identity()
    mx.random.seed(seed)
    model = load_model(str(model_path), dtype=mx.float32)
    model.freeze()
    model.eval()
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )
    if tokenizer.encoding.n_vocab != model.dims.n_vocab:
        raise ValueError("target tokenizer vocabulary does not match model")
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    max_tokens = model.dims.n_text_ctx // 2 if max_tokens is None else max_tokens
    if (
        type(max_tokens) is not int
        or max_tokens <= horizons
        or len(prefix) + max_tokens > model.dims.n_text_ctx
    ):
        raise ValueError(
            "max_tokens must exceed horizons and fit target decoder context "
            "including prefix"
        )
    options = DecodingOptions(
        language=language,
        task=task,
        temperature=0.0,
        sample_len=max_tokens,
        without_timestamps=True,
        fp16=False,
        speculative_mode="none",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    with TemporaryDirectory(prefix=".manifest-", dir=run_dir) as directory:
        candidate = Path(directory) / "manifest.jsonl"
        build_manifest(
            cache_root,
            candidate,
            seed,
            max_samples=2 * model.dims.n_audio_ctx * HOP_LENGTH,
        )
        if (
            manifest_path.exists()
            and candidate.read_bytes() != manifest_path.read_bytes()
        ):
            raise ValueError("cache/split manifest changed; use a new run directory")
        if not manifest_path.exists():
            candidate.replace(manifest_path)
    manifest = read_manifest(manifest_path)
    selected = []
    for split, limit in (
        ("train", max_train_records),
        ("validation", max_validation_records),
    ):
        rows = sorted(
            (row for row in manifest if row["split"] == split),
            key=lambda row: row["audio_digest"],
        )[:limit]
        if not rows:
            raise ValueError(
                f"{split} split is empty; add independent speaker/digest groups"
            )
        selected.extend(rows)
    dataset_digest = _digest(manifest_path)
    tokenizer_digest = fingerprint_tokenizer(tokenizer)
    identity = {
        "schema_version": 2,
        "teacher_fingerprint": model._checkpoint_fingerprint,
        "tokenizer_fingerprint": tokenizer_digest,
        "language": language,
        "task": task,
        "decoder_config": asdict(options),
        "dataset_split_digest": dataset_digest,
        "vocabulary_size": model.dims.n_vocab,
        "prefix": prefix,
        "eot": tokenizer.eot,
        "teacher_source_digests": {
            name: digest
            for name, digest in sources.items()
            if name.startswith("lightning_whisper_mlx/")
        },
    }
    labels_dir = run_dir / "labels"
    if migrate_identity is not None:
        explicit = json.loads(migrate_identity.read_text())
        if explicit != identity:
            raise ValueError(
                "explicit migration identity does not match selected labeling run"
            )
        migrate_legacy_labels(labels_dir, explicit_identity=explicit)
    print(
        f"Labeling {len(selected)} selected clips (train/validation); "
        "source cache read-only",
        flush=True,
    )
    counts = generate_pseudo_labels(
        selected, _teacher(model, options), labels_dir, run_identity=identity
    )
    print(
        f"Labels: {counts['labeled']} completed, {counts['skipped']} resumed, "
        f"{counts['failed']} failed",
        flush=True,
    )
    if counts["failed"]:
        raise ValueError(
            f"teacher labeling failed for {counts['failed']} clips; "
            f"inspect {labels_dir / 'failures.jsonl'} and regenerate corrected "
            "inputs in a new run directory"
        )
    labels_path = labels_dir / "labels.jsonl"
    trainer = MTPTrainer(
        model.dims.n_text_state,
        model.dims.n_vocab,
        horizons,
        learning_rate,
        output_projection=model.decoder.token_embedding.as_linear,
    )
    # Check held-out usability before spending an epoch on training.
    initial_validation = _validation(
        trainer,
        _feature_batches(
            model, selected, labels_path, horizons, batch_size, "validation"
        ),
    )
    history = []
    rng = np.random.default_rng(seed)
    training_records = [row for row in selected if row["split"] == "train"]
    for epoch in range(epochs):
        ordered = [
            training_records[index] for index in rng.permutation(len(training_records))
        ]
        loss_sum, examples = 0.0, 0
        for hidden, targets in _feature_batches(
            model, ordered, labels_path, horizons, batch_size, "train"
        ):
            size = targets.shape[0]
            loss_sum += trainer.train_step(hidden, targets) * size
            examples += size
        if examples == 0:
            raise ValueError(
                "train split has no usable horizon windows; "
                "add clips or reduce horizons"
            )
        metrics = _validation(
            trainer,
            _feature_batches(
                model, selected, labels_path, horizons, batch_size, "validation"
            ),
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_examples": examples,
                "training_loss": loss_sum / examples,
                "validation": metrics,
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs}: {examples} train windows, "
            f"validation loss {metrics['loss']:.6f}, "
            f"prefix acceptance {metrics['teacher_forced_prefix_acceptance']:.4f}",
            flush=True,
        )
    if _source_identity()[1] != sources:
        raise ValueError(
            "training source files changed during run; checkpoint was not exported"
        )
    config = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_train_records": max_train_records,
        "max_validation_records": max_validation_records,
        "label_run_identity": identity,
        "source_digests": sources,
        "selected_audio_digests": {
            split: sorted(
                {row["audio_digest"] for row in selected if row["split"] == split}
            )
            for split in ("train", "validation")
        },
        "initial_validation": initial_validation,
        "epoch_history": history,
    }
    trainer.save_checkpoint(
        checkpoint,
        base_model_repository=str(model_path),
        base_model_revision=model._checkpoint_fingerprint,
        base_model_fingerprint=model._checkpoint_fingerprint,
        tokenizer_fingerprint=tokenizer_digest,
        label_manifest_digest=_digest(labels_path),
        dataset_split_digest=dataset_digest,
        training_commit=commit,
        validation_metrics=metrics,
        training_config=config,
    )
    loaded, metadata = load_mtp_head(
        str(checkpoint),
        decoder_hidden_size=model.dims.n_text_state,
        vocabulary_size=model.dims.n_vocab,
        base_model_fingerprint=model._checkpoint_fingerprint,
        tokenizer_fingerprint=tokenizer_digest,
    )
    reloaded = MTPTrainer(
        model.dims.n_text_state,
        model.dims.n_vocab,
        horizons,
        learning_rate,
        output_projection=model.decoder.token_embedding.as_linear,
    )
    reloaded.head = loaded
    loaded_metrics = _validation(
        reloaded,
        _feature_batches(
            model, selected, labels_path, horizons, batch_size, "validation"
        ),
    )
    if loaded_metrics != metrics:
        raise ValueError("checkpoint reload validation differs from exported head")
    print(
        f"Exported and reloaded {checkpoint}; validation remains held-out", flush=True
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="external read-only NPY cache containing speech/shard_*/__paths__.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="existing local target config.json/weights.npz directory",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="manifest and immutable-identity labeling ledgers",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="new immutable checkpoint directory",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument("--horizons", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=128)
    parser.add_argument("--max-validation-records", type=int, default=32)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--migrate-labels-with-identity",
        type=Path,
        help="explicit attestation of existing legacy label provenance; "
        "backups made before rewrite",
    )
    args = parser.parse_args(argv)
    try:
        run_training(
            cache_root=args.cache_root,
            model_path=args.model,
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            language=args.language,
            task=args.task,
            horizons=args.horizons,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            max_tokens=args.max_tokens,
            migrate_identity=args.migrate_labels_with_identity,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"training failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
