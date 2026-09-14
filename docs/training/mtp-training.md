# MTP training

`training/manifest.py` streams shard metadata and validates NPY files with memory
mapping. `training/pseudo_label.py` writes resumable labels and a failure ledger.
`training/dataset.py` creates deterministic teacher-forced horizon windows.
`training/train_mtp.py` uses `nn.value_and_grad`, MLX Adam, and explicit `mx.eval`.

The target decoder supplies the immediate next token. Head horizon zero predicts
the second future token, horizon one the third, and so on. For prefix `[1]` in
teacher sequence `[1, 2, 3, 4]`, a two-horizon head trains against `[3, 4]`.
Checkpoints trained with the former one-token-shifted labels need retraining before
benchmark qualification. Existing checkpoint files are not modified automatically.

Split allocation targets 80/10/10 from zero assigned records. Speakers and duplicate
digests stay together transitively, so ratios are approximate for indivisible
groups. Missing, empty, and unknown speaker identities do not create a shared
speaker group; duplicate digests still group those records. Unknown identity cannot
prove speaker-disjoint evaluation: verify provenance before qualification. Inspect
actual counts before starting a training run.

Source audio remains external and read-only. Full-cache training requires an
approved local `mlx_audio_cache`, teacher model, split digest, and sufficient disk
space; no such run is claimed by this repository change.

Teacher labeling appends completed records in batches of 32 by default, with a
final flush on normal exit or interruption. `flush_every` can select a different
batch size. Restarting skips persisted labels and failures. Abrupt process death
can lose the current buffered batch or interrupt a ledger write; recovery from a
truncated final JSONL record is not implemented. Keep a backup before a long run.
Each invocation processes new records in deterministic digest order.

`MTPTrainer.save_checkpoint` exports inference weights and compatibility metadata
into a new directory. Both files are prepared in a sibling temporary directory,
then published together with a directory rename. Existing destinations are refused;
use a new path for each export. Failed exports leave previous checkpoints unchanged.
These exports do not include optimizer state and cannot resume training. Optimizer
resume remains unimplemented in this foundation.
