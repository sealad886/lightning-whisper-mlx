from pathlib import Path
import pytest

from lightning_whisper_mlx.cli import (
    STDOUT_OUTPUT,
    discover_media,
    extract_audio,
    build_parser,
    normalize_output_dir,
    output_path_for,
    run_transcription,
    transcribe_file,
)


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_discover_media_supports_files_lists_and_recursive_directories(tmp_path):
    first = touch(tmp_path / "one.mp3")
    second = touch(tmp_path / "nested" / "two.mp4")
    ignored = touch(tmp_path / "nested" / "notes.txt")

    assert discover_media([first]) == [first]
    assert discover_media([first, second]) == [first, second]
    assert discover_media([tmp_path], recursive=False) == [first]
    assert discover_media([tmp_path], recursive=True) == [first, second]
    assert ignored not in discover_media([tmp_path], recursive=True)


def test_output_defaults_beside_source(tmp_path):
    source = tmp_path / "nested" / "recording.m4a"

    assert normalize_output_dir(None) is None
    assert output_path_for(source, None) == source.with_suffix(".txt")


def test_empty_output_dir_is_stdout_sentinel():
    assert normalize_output_dir("") == STDOUT_OUTPUT
    assert normalize_output_dir("''") == STDOUT_OUTPUT
    assert normalize_output_dir('""') == STDOUT_OUTPUT
    assert normalize_output_dir(normalize_output_dir("")) is STDOUT_OUTPUT


def test_output_directory_preserves_relative_structure(tmp_path):
    source = tmp_path / "input" / "a" / "same.wav"
    output_dir = tmp_path / "output"

    assert output_path_for(source, output_dir, relative_to=tmp_path / "input") == (
        output_dir / "a" / "same.txt"
    )


def test_video_audio_extraction_uses_ffmpeg_and_returns_wav(tmp_path, monkeypatch):
    source = touch(tmp_path / "clip.mp4")
    commands = []
    monkeypatch.setattr("lightning_whisper_mlx.cli.shutil.which", lambda _: "/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        Path(command[-1]).write_bytes(b"wav")

    audio = extract_audio(source, run_process=fake_run)

    assert audio.suffix == ".wav"
    assert commands[0][0][:3] == ["/ffmpeg", "-y", "-i"]
    assert commands[0][0][-1] == str(audio)
    audio.unlink()


def test_transcribe_file_cleans_video_temporary_audio(tmp_path, monkeypatch):
    source = touch(tmp_path / "clip.mp4")
    temporary = tmp_path / "temporary.wav"
    temporary.write_bytes(b"wav")
    monkeypatch.setattr(
        "lightning_whisper_mlx.cli.extract_audio", lambda *args, **kwargs: temporary
    )

    text = transcribe_file(source, lambda path: {"text": f"read {path.name}"})

    assert text == "read temporary.wav"
    assert not temporary.exists()


def test_run_transcription_writes_outputs_and_continues_after_failure(tmp_path):
    first = touch(tmp_path / "a.wav")
    second = touch(tmp_path / "b.wav")
    output_dir = tmp_path / "out"
    calls = []

    def transcriber(path):
        calls.append(path)
        if path == second:
            raise RuntimeError("broken input")
        return {"text": "héllo 世界"}

    failures = run_transcription([first, second], transcriber, output_dir=output_dir)

    assert (output_dir / "a.txt").read_text(encoding="utf-8") == "héllo 世界\n"
    assert calls == [first, second]
    assert [(path, str(error)) for path, error in failures] == [
        (second, "broken input")
    ]


def test_run_transcription_prints_distinguishable_stdout_records(tmp_path):
    first = touch(tmp_path / "a.wav")
    second = touch(tmp_path / "b.wav")
    printed = []

    failures = run_transcription(
        [first, second],
        lambda path: {"text": path.stem},
        output_dir="",
        print_fn=lambda message, **kwargs: printed.append(message),
    )

    assert failures == []
    assert printed == [
        "--- " + str(first) + " ---",
        "a",
        "--- " + str(second) + " ---",
        "b",
    ]


def test_parser_has_safe_batch_transcription_defaults():
    args = build_parser().parse_args(["input.wav"])

    assert args.recursive is False
    assert args.output_dir is None
    assert args.model == "tiny"
    assert args.quant is None
    assert args.batch_size == 12


@pytest.mark.parametrize("output_dir", [None, "out"])
def test_colliding_transcripts_fail_without_overwriting(tmp_path, output_dir):
    sources = [touch(tmp_path / "clip.wav"), touch(tmp_path / "clip.mp4")]
    destination_root = tmp_path if output_dir is None else tmp_path / output_dir
    destination_root.mkdir(exist_ok=True)
    transcript = destination_root / "clip.txt"
    transcript.write_text("preserve me")
    calls = []

    failures = run_transcription(
        sources,
        lambda path: calls.append(path),
        output_dir=None if output_dir is None else destination_root,
    )

    assert len(failures) == 2
    assert not calls
    assert transcript.read_text() == "preserve me"


def test_temporary_audio_removed_after_transcription_failure(tmp_path, monkeypatch):
    temporary = touch(tmp_path / "temporary.wav")
    monkeypatch.setattr(
        "lightning_whisper_mlx.cli.extract_audio", lambda *args, **kwargs: temporary
    )

    def fail(path):
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        transcribe_file(tmp_path / "video.mp4", fail)
    assert not temporary.exists()


def test_main_passes_string_path_to_audio_api(tmp_path, monkeypatch):
    from lightning_whisper_mlx import cli
    from lightning_whisper_mlx import lightning

    source = touch(tmp_path / "recording.wav")
    calls = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            calls.append(path)
            return {"text": "hello"}

    monkeypatch.setattr(lightning, "LightningWhisperMLX", Model)
    assert cli.main([str(source)]) == 0
    assert calls == [str(source)]
    assert source.with_suffix(".txt").read_text() == "hello\n"


def test_main_reports_missing_input(tmp_path, capsys):
    from lightning_whisper_mlx.cli import main

    assert main([str(tmp_path / "missing.wav")]) == 2
    assert "does not exist" in capsys.readouterr().err
