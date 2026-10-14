# Transcription CLI Implementation Plan

> **For agentic workers:** Execute task-by-task with TDD and preserve unrelated working-tree changes.

**Goal:** Add a console CLI for single files, multiple files, and recursive directory transcription, including video audio extraction.

**Architecture:** `lightning_whisper_mlx/cli.py` owns media discovery, output planning, ffmpeg extraction, and orchestration around an injected transcription callable. The existing `transcribe_audio` API remains unchanged. `setup.py` exposes `lightning-whisper-mlx` as a console script.

**Tech Stack:** Python standard library (`argparse`, `pathlib`, `subprocess`, `tempfile`), pytest, existing MLX transcription API.

---

### Task 1: Media discovery and output planning

**Files:**
- Create: `lightning_whisper_mlx/cli.py`
- Test: `tests/test_cli.py`

- [ ] Write failing tests for single-file discovery, multi-file discovery, recursive directories, unsupported-file filtering, adjacent output, stdout sentinel values, and relative output-directory mapping.
- [ ] Run focused tests and confirm missing CLI symbols.
- [ ] Implement `SUPPORTED_AUDIO_EXTENSIONS`, `SUPPORTED_VIDEO_EXTENSIONS`, `discover_media`, `output_path_for`, and `normalize_output_dir`.
- [ ] Preserve input order, deduplicate resolved paths, and use `.txt` output names.
- [ ] Run focused tests and confirm green.

### Task 2: Video extraction and transcription orchestration

**Files:**
- Modify: `lightning_whisper_mlx/cli.py`
- Test: `tests/test_cli.py`

- [ ] Write failing tests for ffmpeg command construction, temporary-file cleanup, injected transcription, stdout formatting, output writing, continuation after failure, and nonzero failure count.
- [ ] Run focused tests and confirm failure against placeholders.
- [ ] Implement `extract_audio`, `transcribe_file`, and `run_transcription` with injectable `runner` and `subprocess.run` boundary.
- [ ] Use ffmpeg only for video inputs; raise an actionable `RuntimeError` when unavailable or extraction fails.
- [ ] Continue batch processing while collecting failures.
- [ ] Run focused tests and confirm green.

### Task 3: CLI parser, console script, and documentation

**Files:**
- Modify: `lightning_whisper_mlx/cli.py`
- Modify: `setup.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

- [ ] Write failing tests for parser defaults, `--recursive`, `--output-dir`, model/quant/batch options, and CLI exit codes.
- [ ] Add `main(argv=None)` and `console_scripts` entrypoint.
- [ ] Wire the default runner to `LightningWhisperMLX.transcribe`, while tests inject a fake runner.
- [ ] Document file, multi-file, recursive directory, stdout, output-directory, and video usage.
- [ ] Run full tests, compileall, focused lint, and diff checks.
