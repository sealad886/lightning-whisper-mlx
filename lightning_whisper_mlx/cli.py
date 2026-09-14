"""Command-line transcription for audio and video media."""

from __future__ import annotations

import argparse
from collections import Counter
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

STDOUT_OUTPUT = object()

SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".aiff", ".opus"}
)
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
)
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


def is_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def discover_media(
    inputs: Iterable[Path | str], *, recursive: bool = False
) -> list[Path]:
    """Discover supported media in input order without duplicate paths."""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_file():
            candidates = [path] if is_media(path) else []
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = sorted(
                path.glob(pattern),
                key=lambda candidate: (
                    len(candidate.relative_to(path).parts),
                    str(candidate),
                ),
            )
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if is_media(candidate) and resolved not in seen:
                seen.add(resolved)
                discovered.append(candidate)
    return discovered


def normalize_output_dir(value: Optional[str | Path]):
    """Return None for adjacent output, a sentinel for stdout, or a Path."""
    if value is None:
        return None
    text = str(value)
    if text in {"", "''", '""'}:
        return STDOUT_OUTPUT
    return Path(value)


def output_path_for(
    source: Path,
    output_dir: Optional[Path],
    *,
    relative_to: Optional[Path] = None,
):
    if output_dir is None:
        return source.with_suffix(".txt")
    if output_dir is STDOUT_OUTPUT:
        return None
    relative = (
        source.name
        if relative_to is None
        else source.relative_to(relative_to).with_suffix(".txt")
    )
    if isinstance(relative, str):
        relative = Path(relative).with_suffix(".txt")
    return output_dir / relative


def extract_audio(
    source: Path,
    *,
    run_process: Callable = subprocess.run,
) -> Path:
    """Extract a normalized mono WAV from a video into a temporary file."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "video transcription requires ffmpeg; install ffmpeg and retry"
        )
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    output = Path(temporary.name)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output),
    ]
    try:
        try:
            run_process(command, check=True, capture_output=True)
        except FileNotFoundError as error:
            raise RuntimeError(
                "video transcription requires ffmpeg; install ffmpeg and retry"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode(errors="replace") if error.stderr else ""
            raise RuntimeError(f"ffmpeg could not extract audio: {detail}") from error
        return output
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _text_from_result(result) -> str:
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)


def transcribe_file(
    source: Path,
    transcriber: Callable[[Path], object],
    *,
    run_process: Callable = subprocess.run,
) -> str:
    temporary: Optional[Path] = None
    input_path = source
    try:
        if source.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            temporary = extract_audio(source, run_process=run_process)
            input_path = temporary
        return _text_from_result(transcriber(input_path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _common_root(sources: Sequence[Path]) -> Path:
    parents = [source.parent.resolve() for source in sources]
    return Path(os.path.commonpath([str(parent) for parent in parents]))


def run_transcription(
    sources: Sequence[Path],
    transcriber: Callable[[Path], object],
    *,
    output_dir=None,
    relative_to: Optional[Path] = None,
    run_process: Callable = subprocess.run,
    print_fn: Callable[..., None] = print,
) -> list[tuple[Path, Exception]]:
    """Transcribe all sources, continuing after failures and returning failures."""
    normalized_output = normalize_output_dir(output_dir)
    if not sources:
        return []
    root = relative_to or _common_root(sources)
    destinations = [
        output_path_for(source.resolve(), normalized_output, relative_to=root)
        for source in sources
    ]
    counts = Counter(path for path in destinations if path is not None)
    failures: list[tuple[Path, Exception]] = []
    for source, destination in zip(sources, destinations):
        try:
            if destination is not None and counts[destination] > 1:
                raise ValueError(
                    f"multiple inputs map to {destination}; rename inputs or use stdout"
                )
            text = transcribe_file(source, transcriber, run_process=run_process)
            if destination is None:
                print_fn(f"--- {source} ---")
                print_fn(text)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    text + ("\n" if text and not text.endswith("\n") else ""),
                    encoding="utf-8",
                )
        except Exception as error:
            failures.append((source, error))
            print_fn(f"error: {source}: {error}", file=sys.stderr)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--quant", choices=["4bit", "8bit"], default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--language", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        print("batch size must be positive", file=sys.stderr)
        return 2
    try:
        sources = discover_media(args.inputs, recursive=args.recursive)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not sources:
        print("no supported media files found", file=sys.stderr)
        return 2

    from .lightning import LightningWhisperMLX

    model = LightningWhisperMLX(
        model=args.model,
        batch_size=args.batch_size,
        quant=args.quant,
    )

    def transcriber(path):
        return model.transcribe(str(path), language=args.language)

    failures = run_transcription(
        sources,
        transcriber,
        output_dir=args.output_dir,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
