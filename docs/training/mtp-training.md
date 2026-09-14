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
groups. Inspect actual counts before starting a training run.

Source audio remains external and read-only. Full-cache training requires an
approved local `mlx_audio_cache`, teacher model, split digest, and sufficient disk
space; no such run is claimed by this repository change.
