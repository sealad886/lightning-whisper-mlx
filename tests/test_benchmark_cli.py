import json

import pytest

from benchmarks import benchmark_decoding


def test_cli_loads_explicit_runner_and_writes_json(tmp_path, monkeypatch):
    module = tmp_path / "fixture_runner.py"
    module.write_text(
        "def run(case):\n"
        "    return {'tokens': [1, 2, 3]}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "result.json"

    benchmark_decoding.main(
        [
            "--runner",
            "fixture_runner:run",
            "--output",
            str(output),
            "--warmups",
            "0",
            "--repetitions",
            "2",
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["results"][0]["case"]["mode"] == "none"
    assert payload["results"][0]["repetitions"] == 2
    assert payload["results"][0]["token_count"] == 3


def test_cli_requires_explicit_runner(tmp_path):
    with pytest.raises(SystemExit) as error:
        benchmark_decoding.main(["--output", str(tmp_path / "result.json")])

    assert error.value.code == 2
