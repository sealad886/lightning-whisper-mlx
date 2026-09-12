# MTP training

`training/manifest.py` streams shard metadata and validates NPY files with memory
mapping. `training/pseudo_label.py` writes resumable labels and a failure ledger.
`training/dataset.py` creates deterministic teacher-forced horizon windows.
`training/train_mtp.py` uses `nn.value_and_grad`, MLX Adam, and explicit `mx.eval`.

Source audio remains external and read-only. Full-cache training requires an
approved local `mlx_audio_cache`, teacher model, split digest, and sufficient disk
space; no such run is claimed by this repository change.
