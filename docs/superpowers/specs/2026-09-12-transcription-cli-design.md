# Transcription CLI Design

**Status:** Approved

## Goal

Provide one CLI for single-file, multi-file, and recursive directory transcription.
Audio and video media share one orchestration path; video audio is extracted with
`ffmpeg` into a temporary WAV first.

## Output contract

Without `--output-dir`, write `<source-stem>.txt` beside each source. With an empty
output-dir value (`''` or `""`), print each transcript to stdout with a source header.
With a non-empty output directory, preserve each source's relative parent structure
and write one `.txt` per source.

## Failure contract

Unsupported files are ignored during directory discovery. Missing inputs fail
immediately. Per-file transcription failures are reported while remaining files
continue; the CLI exits nonzero if any file fails.

## Boundaries

The CLI injects the transcription callable and subprocess runner in tests. The
production callable creates one `LightningWhisperMLX` model and transcribes each
input. Video extraction requires `ffmpeg` on `PATH`; temporary files are removed in
all success and failure paths.
