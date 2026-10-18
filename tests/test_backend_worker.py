"""Protocol tests use a tiny CPU-only process, never an MLX/Metal workload."""

import struct
import sys

import pytest

from benchmarks.cpp_backend import CppBackend, MAX_RESPONSE, build_worker


@pytest.fixture
def worker(tmp_path):
    script = tmp_path / "worker"
    script.write_text(f"#!{sys.executable}\n" + """
import json, struct, sys

def reply(value):
    data=json.dumps(value).encode(); sys.stdout.buffer.write(struct.pack('<I',len(data))+data); sys.stdout.buffer.flush()
reply({'ready':True})
while True:
    op=sys.stdin.buffer.read(1)
    if not op: break
    op=op[0]
    if op==3: break
    clip_id,=struct.unpack('<I',sys.stdin.buffer.read(4))
    if op==1:
        count,=struct.unpack('<I',sys.stdin.buffer.read(4))
        assert len(sys.stdin.buffer.read(count*4))==count*4
        reply({'uploaded':True})
    elif op==2:
        reply({'text':str(clip_id),'segments':[],'tokens':[],'language':'en'})
""")
    script.chmod(0o700)
    return script


def test_persistent_upload_repeated_requests_and_cleanup(worker):
    with CppBackend(worker, "unused", language="en") as backend:
        process = backend.process
        backend.upload(7, struct.pack("<ff", 0.25, -0.5))
        assert backend.transcribe(7)["text"] == "7"
        assert backend.transcribe(7)["text"] == "7"
        with pytest.raises(ValueError, match="duplicate"):
            backend.upload(7, b"1234")
        with pytest.raises(ValueError, match="not uploaded"):
            backend.transcribe(8)
    assert process.poll() is not None
    assert process.stdout.closed and process.stdin.closed


@pytest.mark.parametrize("pcm", [b"", b"a", b"12345"])
def test_rejects_malformed_pcm(worker, pcm):
    with CppBackend(worker, "unused", language="en") as backend:
        with pytest.raises(ValueError, match="PCM"):
            backend.upload(0, pcm)


def test_init_timeout_kills_worker(tmp_path):
    script = tmp_path / "stuck"
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    script.chmod(0o700)
    with pytest.raises(TimeoutError):
        CppBackend(script, "unused", language="en", timeout=0.05)


def test_response_size_guard(tmp_path):
    script = tmp_path / "bad"
    script.write_text(
        f"#!{sys.executable}\nimport sys,struct\nsys.stdout.buffer.write(struct.pack('<I',{MAX_RESPONSE+1}));sys.stdout.buffer.flush()\n"
    )
    script.chmod(0o700)
    with pytest.raises(RuntimeError, match="size"):
        CppBackend(script, "unused", language="en")


def test_build_rejects_non_metal_or_debug(tmp_path):
    source = tmp_path / "source"
    (source / "include").mkdir(parents=True)
    (source / "include/whisper.h").write_text("header")
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\nGGML_METAL:BOOL=OFF\nCMAKE_BUILD_TYPE:STRING=Release\n"
    )
    with pytest.raises(ValueError, match="Metal Release"):
        build_worker(source, build, tmp_path / "worker")


def test_worker_receives_explicit_sorted_suppression_ids(worker):
    # Existing fake worker ignores argv, so inspect the actual spawned command.
    with CppBackend(
        worker, "unused", language="en", suppress_tokens=(3268, 4, 4)
    ) as backend:
        assert backend.process.args[-1] == "4,3268"
    with pytest.raises(ValueError, match="suppression IDs"):
        CppBackend(worker, "unused", language="en", suppress_tokens=(-1,))
