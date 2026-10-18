"""CLI collision checks use filesystem identities before transcribing inputs."""

from pathlib import Path
import runpy

import pytest

cli = runpy.run_path(
    str(Path(__file__).parents[1] / "lightning_whisper_mlx" / "cli.py")
)


@pytest.mark.parametrize("names", [("clip", "CLIP"), ("caf\u00e9", "cafe\u0301")])
def test_filesystem_aliases_cannot_overwrite_transcripts(tmp_path, names):
    first, second = [
        tmp_path / (stem + ext) for stem, ext in zip(names, (".wav", ".flac"))
    ]
    first.touch()
    second.touch()
    # Require an actual aliasing filesystem for this integration scenario.
    probe = tmp_path / (names[0] + ".probe")
    probe.touch()
    if not (tmp_path / (names[1] + ".probe")).exists():
        pytest.skip("filesystem preserves distinct case/normalization names")
    calls = []
    failures = cli["run_transcription"](
        [first, second], lambda path: calls.append(path) or path.name
    )
    assert len(failures) == 2
    assert not calls
    assert not first.with_suffix(".txt").exists()


def test_unwritable_destination_preflight_does_not_block_other_inputs(
    tmp_path, monkeypatch
):
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    first.touch()
    second.touch()
    calls = []
    original = Path.stat

    def stat(path, *args, **kwargs):
        if path == first.with_suffix(".txt"):
            raise PermissionError("destination inaccessible")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    failures = cli["run_transcription"](
        [first, second], lambda path: calls.append(path) or path.name
    )
    assert [path for path, _ in failures] == [first]
    assert calls == [second]
    assert second.with_suffix(".txt").read_text().strip() == second.name
