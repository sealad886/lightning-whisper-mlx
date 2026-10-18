"""Offline persistent whisper.cpp adapter; no MLX import or model downloads.

Protocol: byte opcode (1 upload, 2 transcribe, 3 quit), little-endian uint32
clip id, then upload-only uint32 sample count and float32 mono 16 kHz PCM.
Responses are little-endian uint32 length followed by UTF-8 JSON. PCM is uploaded
before timing, so both warmed backends consume resident canonical audio.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import selectors
import struct
import subprocess
import tempfile
import time

MAX_SAMPLES = 16000 * 60 * 30
MAX_RESPONSE = 64 * 1024 * 1024


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(source):
    """Fingerprint tracked build inputs, including dirty bytes, rather than HEAD alone."""
    source = Path(source).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    digest = hashlib.sha256()
    files = git("ls-files", "-z").split(b"\0")
    newest = 0
    for raw in sorted(filter(None, files)):
        path = source / os.fsdecode(raw)
        if path.is_file() and (
            path.suffix in {".h", ".hpp", ".c", ".cpp", ".m", ".mm", ".metal", ".cmake"}
            or path.name == "CMakeLists.txt"
        ):
            digest.update(raw + b"\0" + file_digest(path).encode())
            newest = max(newest, path.stat().st_mtime_ns)
    return {
        "path": str(source),
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain")),
        "build_inputs_sha256": digest.hexdigest(),
        "newest_input_mtime_ns": newest,
    }


def build_worker(source, build, output):
    """Compile worker against a selected existing local CMake build, recording identity.

    A library with newer source inputs is rejected. For strongest provenance build
    whisper.cpp out of tree immediately before this command and retain its receipt.
    No source checkout is modified by this function.
    """
    source, build, output = map(lambda p: Path(p).resolve(), (source, build, output))
    if not (source / "include/whisper.h").is_file():
        raise ValueError("missing public whisper.h")
    cache = build / "CMakeCache.txt"
    cache_text = cache.read_text()
    home = "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(source)
    if (
        home not in cache_text.splitlines()
        and ("LIGHTNING_WHISPER_CPP_SOURCE:PATH=" + str(source))
        not in cache_text.splitlines()
    ):
        raise ValueError("CMake build belongs to another source checkout")
    for required in ("GGML_METAL:BOOL=ON", "CMAKE_BUILD_TYPE:STRING=Release"):
        if required not in cache_text:
            raise ValueError(f"optimized Metal Release build required: {required}")
    if "WHISPER_COREML:BOOL=ON" in cache_text:
        raise ValueError(
            "Core ML must be qualified as a separate backend configuration"
        )
    candidates = sorted((build / "bin").glob("libwhisper.*.dylib"))
    if not candidates:
        candidates = sorted((build / "src").glob("libwhisper*.dylib"))
    if not candidates:
        raise ValueError("missing built libwhisper dylib")
    library = candidates[0].resolve()
    identity = source_identity(source)
    if identity["newest_input_mtime_ns"] > library.stat().st_mtime_ns:
        raise ValueError(
            "whisper.cpp build is older than source inputs; rebuild out of tree"
        )
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("worker output/receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    worker_source = Path(__file__).with_name("whisper_cpp_worker.cpp")
    command = [
        "clang++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-I",
        str(source / "include"),
        "-I",
        str(source / "ggml/include"),
        str(worker_source),
        str(library),
        "-Wl,-rpath," + str(library.parent),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    dependencies = {
        str(p.resolve()): file_digest(p) for p in library.parent.glob("*.dylib")
    }
    for dependency in re.findall(
        r"^[^#\n][^\n]*:FILEPATH=(/[^\n]+\.dylib)$", cache_text, re.MULTILINE
    ):
        path = Path(dependency).resolve()
        if path.is_file():
            dependencies[str(path)] = file_digest(path)
    receipt = {
        "source": identity,
        "library": str(library),
        "libraries": dependencies,
        "cmake_cache_sha256": file_digest(cache),
        "cmake_cache": cache_text,
        "worker_source_sha256": file_digest(worker_source),
        "command": command,
        "worker_sha256": file_digest(output),
        "build_provenance": "existing-local-build; mtime-checked; byte-fingerprinted",
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


class CppBackend:
    def __init__(
        self,
        worker,
        model,
        *,
        language,
        threads=4,
        beam_size=1,
        condition=True,
        timestamps=True,
        task="transcribe",
        timeout=300,
        suppress_tokens=(),
    ):
        if (
            threads < 1
            or beam_size < 1
            or not language
            or task not in {"transcribe", "translate"}
        ):
            raise ValueError("invalid backend options")
        if not 0 < timeout <= 3600:
            raise ValueError("timeout must be positive and <=3600 seconds")
        suppressed = tuple(sorted(set(suppress_tokens)))
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= 2**31
            for token in suppressed
        ):
            raise ValueError("suppression IDs must be nonnegative integers")
        suppression_csv = ",".join(map(str, suppressed))
        if len(suppression_csv) > 65536:
            raise ValueError("suppression list exceeds limit")
        self.timeout = timeout
        self.total_samples = 0
        self.ids = set()
        self.log = tempfile.TemporaryFile()
        environment = os.environ.copy()
        # ANE is a separately qualified backend, never an inherited surprise.
        environment.pop("ANEFORGE_ENCODER", None)
        environment.pop("ANEFORGE_DYLIB", None)
        environment["OMP_NUM_THREADS"] = str(threads)
        try:
            self.process = subprocess.Popen(
                [
                    str(worker),
                    str(model),
                    language,
                    str(threads),
                    str(beam_size),
                    str(int(condition)),
                    str(int(timestamps)),
                    task,
                    suppression_csv,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
                env=environment,
            )
        except BaseException:
            self.log.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            self.ready = self._response()
            if self.ready.get("ready") is not True:
                raise RuntimeError("worker did not initialize")
            if self.ready.get("suppression_count", len(suppressed)) != len(suppressed):
                raise RuntimeError("worker suppression configuration differs")
        except BaseException:
            self.close()
            raise

    def _io(self, data=None, count=None):
        stream = self.process.stdin if data is not None else self.process.stdout
        event = selectors.EVENT_WRITE if data is not None else selectors.EVENT_READ
        deadline = time.monotonic() + self.timeout
        result = bytearray()
        position = 0
        with selectors.DefaultSelector() as selector:
            selector.register(stream, event)
            while position < (len(data) if data is not None else count):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("whisper.cpp worker I/O timeout")
                if data is not None:
                    try:
                        moved = os.write(
                            stream.fileno(),
                            memoryview(data)[position : position + 65536],
                        )
                    except BlockingIOError:
                        continue
                else:
                    try:
                        chunk = os.read(stream.fileno(), min(count - position, 65536))
                    except BlockingIOError:
                        continue
                    moved = len(chunk)
                    result.extend(chunk)
                if not moved:
                    raise RuntimeError("whisper.cpp worker pipe closed")
                position += moved
        return bytes(result)

    def _response(self):
        (size,) = struct.unpack("<I", self._io(count=4))
        if size < 2 or size > MAX_RESPONSE:
            raise RuntimeError("invalid worker response size")
        response = json.loads(self._io(count=size))
        if not isinstance(response, dict):
            raise RuntimeError("worker response must be object")
        if "error" in response:
            raise RuntimeError("whisper.cpp: " + str(response["error"]))
        return response

    def upload(self, clip_id, pcm):
        # memoryview keeps this adapter independent of NumPy at import time.
        if clip_id in self.ids or not 0 <= clip_id < 2**32:
            raise ValueError("duplicate/invalid clip id")
        count = len(pcm) // 4
        if len(pcm) % 4 or not count or self.total_samples + count > MAX_SAMPLES:
            raise ValueError("PCM must be float32 and total duration <=30 minutes")
        self._io(data=struct.pack("<BII", 1, clip_id, count) + pcm)
        response = self._response()
        if response.get("uploaded") is not True:
            raise RuntimeError("worker upload was not acknowledged")
        self.ids.add(clip_id)
        self.total_samples += count

    def transcribe(self, clip_id):
        if clip_id not in self.ids:
            raise ValueError("clip was not uploaded")
        self._io(data=struct.pack("<BI", 2, clip_id))
        return self._response()

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if getattr(self, "log", None):
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Build persistent Metal whisper.cpp benchmark worker"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_worker(args.source, args.build, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
