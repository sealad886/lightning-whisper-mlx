"""A functional request must receive the checkpoint it requested."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import mlx.core as mx

from lightning_whisper_mlx import transcribe


def test_concurrent_model_loads_keep_returned_model_and_cache_identity_consistent(
    monkeypatch,
):
    holder = transcribe.ModelHolder
    for attribute in ("model", "model_path", "model_identity"):
        monkeypatch.setattr(holder, attribute, None)

    first_publishing = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    identities = {}
    models = {name: SimpleNamespace(checkpoint=name) for name in ("first", "second")}

    def identity(path):
        identities[path] = identities.get(path, 0) + 1
        if path == "first" and identities[path] == 2:
            # The old implementation has published its model but has not yet
            # published its identity. A competing call can replace that model.
            first_publishing.set()
            if not release_first.wait(timeout=10):
                raise TimeoutError("test did not release the first request")
        return ("artifact", path)

    monkeypatch.setattr(transcribe, "model_artifact_identity", identity)
    monkeypatch.setattr(transcribe, "load_model", lambda path, dtype: models[path])

    def second_request():
        try:
            return holder.get_model("second", mx.float16)
        finally:
            second_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(holder.get_model, "first", mx.float16)
        try:
            assert first_publishing.wait(timeout=5), "first request did not publish"
            second = executor.submit(second_request)
            # With serialized publication, the second request waits for the
            # first. Without it, it finishes before the first is released.
            second_finished.wait(timeout=1)
        finally:
            release_first.set()
        first_model = first.result(timeout=5)
        second_model = second.result(timeout=5)

    assert first_model is models["first"]
    assert second_model is models["second"]
    cached_path = holder.model_path
    assert holder.model is models[cached_path]
    assert holder.model_identity == (("artifact", cached_path), str(mx.float16))
    assert holder.get_model(cached_path, mx.float16) is models[cached_path]
