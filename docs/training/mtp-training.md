# MTP training

Run bounded local training with an approved external NPY cache and an existing
local target model. This command never downloads audio or model weights:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m training.train_mtp \
  --cache-root /path/to/approved/mlx_audio_cache \
  --model /path/to/local/target \
  --run-dir /path/to/training-run \
  --checkpoint /path/to/new-checkpoint \
  --language en --task transcribe --horizons 2 \
  --epochs 1 --batch-size 16 --seed 0 \
  --max-train-records 128 --max-validation-records 32
```

Run outside the sandbox on Apple Silicon because MLX requires Metal access.
`--model` must contain `config.json` and `weights.npz`. Run/checkpoint destinations
must be outside the source cache. The checkpoint destination must not exist.
Defaults bound labeling to 128 training and 32 validation records, one epoch, and
16 hidden-state windows per optimizer step. Each clip's teacher generation is
bounded by `--max-tokens` (default: half the target text context). Increase these
limits only for an intentionally selected run. Manifest creation validates and
hashes all source records; selected-record limits apply to labeling and training.

The cache layout is `speech/shard_*/__paths__.json`, with shard-relative
`audio_path` entries. Audio must be finite, mono, floating-point NPY at 16000 Hz,
nonempty, and no longer than the target encoder context (30 seconds for standard
Whisper). Incorrect sample rate, nonfinite samples, oversized clips, inaccessible
files and metadata paths escaping the shard fail with source-specific errors.
Resample/chunk unsupported audio separately; this workflow does not mutate it.
Manifest `source_path` is an absolute resolved shard directory;
`relative_audio_path` remains shard-relative. Source bytes are rehashed before
labeling and feature extraction to detect changes after manifest creation.

The workflow:

1. Builds a deterministic manifest and grouped 80/10/10 train/validation/test split.
   Speaker identities and duplicate digests stay together transitively. Unknown
   speaker identities do not form one shared group; unknown identities cannot
   establish speaker-disjoint evaluation. Ratios are approximate for indivisible
   groups. Empty train/validation splits stop the run.
2. Selects a deterministic bounded subset from train and validation, leaving test
   records unused. The frozen target produces temperature-zero, text-only labels
   through the canonical decoder and suppression filters. Labels include the exact
   complete decoder prefix and terminal EOT. Generation reaching its token bound
   is rejected rather than inventing successful termination.
3. Encodes each selected clip and teacher-forces the full labeled sequence once
   per pass. The head receives the normalized hidden state at each complete prefix.
   Only head parameters enter `nn.value_and_grad`; target parameters stay frozen.
   The tied `token_embedding.as_linear` projection supports dense and quantized
   targets without unpacking quantized weights as a dense matrix.
4. Checks held-out usability before training, trains bounded batches for the chosen
   epochs, and evaluates validation windows after every epoch. Missing examples or
   nonfinite loss/metrics stop export. Training records never enter validation.
5. Exports into a new directory, reloads with strict model/tokenizer/shape checks,
   and re-evaluates the loaded head on the same held-out windows.

For complete prefix `[1]` in teacher sequence `[1, 2, 3, 4]`, a two-horizon head
trains against `[3, 4]`: the target decoder supplies token `2`, while horizon zero
predicts the second future token. Windows begin only after the complete canonical
prefix, including language/task/no-timestamp controls. Terminal EOT can be a target.

Metadata stores measured held-out loss, per-horizon accuracy, example counts, and
`teacher_forced_prefix_acceptance`: the fraction of head horizons matching an
unbroken teacher-forced target continuation. This is an offline proposal metric;
it does not measure live speculative verifier acceptance or speedup. Qualification
still requires representative held-out runtime parity, acceptance, timing and
stability evidence from the decoding benchmark workflow.

## Identity, resume and migration

`labels/identity.json` binds labels to schema 2, the exact teacher checkpoint,
complete tokenizer fingerprint, language, task, all decoder options, split-manifest
digest, teacher source-code digests, vocabulary size, canonical prefix and EOT.
Every successful/failure record
carries its run digest. Resume rejects mismatched identity before calling the
teacher or appending anything. Semantic checks reject missing tokens, booleans,
non-integer/out-of-vocabulary IDs, incorrect prefixes and premature/missing EOT.
Failure records remain failed on restart; inspect their ledger and use a new run
directory after correcting inputs.

Restarting with the same run directory reuses labels. Training starts from a new
seeded head; exported checkpoints contain no optimizer state and cannot resume an
interrupted optimizer. Use a fresh checkpoint path for each export. Changing
source files, split seed, teacher or decoder configuration requires a new run
directory. Epoch count, head horizons and batch size can change while reusing the
same correctly identified labels.

Legacy ledgers with no identity fail closed. Prefer regeneration in a new run
directory. If provenance is already known and explicitly attested, the once-off
`migrate_legacy_labels(..., explicit_identity=...)` helper, or CLI option
`--migrate-labels-with-identity /path/to/attested-identity.json`, validates existing
labels, creates `legacy-backup-<id>` inside their output directory, and rewrites the
original ledgers with the run digest before creating identity metadata. CLI identity
must exactly match the selected teacher/tokenizer/manifest/decoder run. Migration
never infers missing teacher provenance or inserts missing canonical prefix/EOT;
those old labels must be regenerated. Backups preserve original ledger bytes.

Checkpoint schema 2 fingerprints all ordinary token-byte/ID mappings, all special
tokens and the tokenizer segmentation regex. Schema 1 checkpoints are rejected
with regeneration/retraining guidance; they cannot be qualified by changing the
schema number. Heads trained with the former one-token-shifted targets also require
retraining. No legacy checkpoint adapter is provided.

Labels append in batches of 32 with a final flush on normal exit/interruption.
Abrupt death can lose the current buffered batch or truncate a final JSONL record;
invalid ledgers stop with an actionable restore-backup error. Keep local backups
before a long run; concurrent writers to the same run directory are unsupported.

## Export and evidence

Both inference weights and compatibility metadata are prepared in a sibling
temporary directory, then published together by directory rename. Existing
destinations are refused. Metadata records checkpoint/tokenizer/split/label digests,
selected train/validation audio digests, actual epoch/validation history, seed,
bounds, source-file digests and training Git revision (with `+dirty` when applicable).
Source digests identify the training code used even before its local commit.

The synthetic native integration test runs this same entrypoint against generated
NPY fixtures, a tiny actual Whisper checkpoint and controlled tokenizer. It proves
training updates, frozen target weights, disjoint validation, deterministic restart,
export and reload. It supplies no real-audio quality, full-cache training, runtime
speedup, licensing or production qualification claim. Source audio remains external
and read-only; generated manifests, ledgers and checkpoints should remain outside Git.
