"""Contention decisions use synthetic OS counters and never import MLX."""

import pytest

from benchmarks.contention import (
    ContentionPolicy,
    QuietGuard,
    ContentionError,
    assess_window,
)


def observations():
    return [
        dict(
            timestamp_utc=f"sample-{i}",
            cpu_ticks=[10, 5, 85 + 100 * i, 0],
            gpu_device_percent=1,
            gpu_clients=[
                dict(
                    pid=431,
                    name="WindowServer",
                    gpu_time_raw=i * 10,
                    last_submission_raw=i,
                )
            ],
            thermal_state=0,
            power_source="AC Power",
            memory_free_percent=50,
            vm_counters=dict(pageouts=5, swapins=5, swapouts=5),
            processes=[],
            errors={},
        )
        for i in range(3)
    ]


def test_quiet_window_requires_real_gpu_thermal_and_paging_evidence():
    report = assess_window(observations(), ContentionPolicy())
    assert report["status"] == "observed-low-contention"
    assert report["unknown"] == []
    assert not report["guarantees_uncontended"]
    assert report["external_gpu_activity"] == []


def test_external_gpu_work_is_rejected_even_when_cpu_idle():
    values = observations()
    for i, value in enumerate(values):
        value["gpu_clients"].append(
            dict(pid=9, name="other mlx", gpu_time_raw=i * 5, last_submission_raw=i)
        )
    report = assess_window(values, ContentionPolicy())
    assert report["status"] == "busy"
    assert report["external_gpu_activity"][0]["pid"] == 9
    assert (
        assess_window(values, ContentionPolicy(), owned_pids={9})["status"]
        == "observed-low-contention"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("thermal_state", 1),
        ("gpu_device_percent", 40),
        ("power_source", "Battery Power"),
        ("memory_free_percent", 1),
    ],
)
def test_bad_environment_is_busy(field, value):
    values = observations()
    values[1][field] = value
    assert assess_window(values, ContentionPolicy())["status"] == "busy"


def test_unknown_gpu_and_thermal_cannot_pass_from_cpu_alone():
    values = observations()
    values[1]["thermal_state"] = None
    values[2]["gpu_device_percent"] = None
    report = assess_window(values, ContentionPolicy())
    assert report["status"] == "unknown"
    assert report["unknown"]


def test_paging_and_heavy_compute_cpu_reject_window():
    values = observations()
    values[-1]["vm_counters"]["swapouts"] += 1
    values[1]["processes"] = [dict(pid=20, cpu_percent=55, comm="graphify/python")]
    report = assess_window(values, ContentionPolicy())
    assert report["status"] == "busy"
    assert any("paging" in reason for reason in report["reasons"])
    assert any("compute" in reason for reason in report["reasons"])


def test_strict_guard_keeps_failure_receipt_and_bounds_sampling():
    values = observations()
    values[0]["gpu_device_percent"] = 99
    pending = iter(values)
    sleeps = []
    guard = QuietGuard(
        ContentionPolicy(),
        required=True,
        collector=lambda: next(pending),
        sleeper=sleeps.append,
    )
    with pytest.raises(ContentionError):
        guard.check("before-pair", cooldown=True)
    assert len(guard.checks) == 1
    assert len(guard.checks[0]["observations"]) == 3
    assert guard.checks[0]["status"] == "busy"
    assert sleeps == [2.0, 0.5, 0.5]


def test_diagnostic_guard_records_unknown_without_claiming_requirement():
    values = observations()
    values[0]["gpu_clients"] = None
    pending = iter(values)
    guard = QuietGuard(
        ContentionPolicy(), collector=lambda: next(pending), sleeper=lambda _: None
    )
    guard.check("preflight")
    assert not guard.summary()["qualified"]
    assert guard.summary()["status"] == "unknown"


def test_gpu_counters_cover_timed_gap_without_counting_owned_work():
    first, second = observations(), observations()
    for sample in first:
        sample["gpu_clients"].append(
            dict(pid=9, name="external", gpu_time_raw=0, last_submission_raw=0)
        )
    for sample in second:
        sample["gpu_clients"][0]["gpu_time_raw"] += 30
        sample["gpu_clients"][0]["last_submission_raw"] += 3
        sample["gpu_clients"].append(
            dict(pid=9, name="external", gpu_time_raw=500, last_submission_raw=1)
        )
    pending = iter(first + second)
    guard = QuietGuard(collector=lambda: next(pending), sleeper=lambda _: None)
    guard.check("before")
    after = guard.check("after")
    assert after["status"] == "busy"
    assert after["between_checks_gpu_activity"][0]["pid"] == 9
    pending = iter(first + second)
    owned = QuietGuard(
        owned_pids={9}, collector=lambda: next(pending), sleeper=lambda _: None
    )
    owned.check("before")
    assert owned.check("after")["status"] == "observed-low-contention"


def test_owned_gpu_counter_lifecycle_is_not_external_contention():
    values = observations()
    for sample, counter in zip(values, (100, 0, 5)):
        sample["gpu_clients"].append(
            dict(pid=9, name="benchmark", gpu_time_raw=counter, last_submission_raw=1)
        )
    assert (
        assess_window(values, ContentionPolicy(), owned_pids={9})["status"]
        == "observed-low-contention"
    )


def test_paging_during_timed_gap_invalidates_observed_quiet_run():
    first, second = observations(), observations()
    for sample in second:
        sample["gpu_clients"][0]["gpu_time_raw"] += 30
        sample["gpu_clients"][0]["last_submission_raw"] += 3
        sample["vm_counters"]["swapins"] += 1
    pending = iter(first + second)
    guard = QuietGuard(collector=lambda: next(pending), sleeper=lambda _: None)
    guard.check("before")
    report = guard.check("after")
    assert report["status"] == "busy"
    assert any(
        "between checks: active paging" in reason for reason in report["reasons"]
    )


@pytest.mark.parametrize("status,exit_code", [("unknown", 3), ("busy", 4)])
def test_required_cli_writes_failure_before_importing_mlx(
    monkeypatch, tmp_path, status, exit_code
):
    import builtins
    import json
    import numpy as np
    from benchmarks import compare_backends

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"test-audio")
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        compare_backends, "validate_artifacts", lambda args: ({}, {"test": True})
    )
    monkeypatch.setattr(
        compare_backends, "decode_pcm", lambda path: np.zeros(1600, dtype=np.float32)
    )

    class Guard:
        def __init__(self, policy, required=False):
            assert required

        def check(self, phase, **kwargs):
            assert phase == "preflight-before-model-loading"
            raise ContentionError("known preflight rejection")

        def summary(self):
            return {
                "status": status,
                "qualified": False,
                "checks": [{"status": status}],
            }

    monkeypatch.setattr(compare_backends, "QuietGuard", Guard)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert not name.startswith(
            "mlx"
        ), "preflight failure must precede MLX import/model initialization"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert (
        compare_backends.main(
            [
                "--mlx-model",
                "unused",
                "--cpp-model",
                "unused",
                "--worker",
                "unused",
                "--checkpoint",
                "base",
                "--audio",
                str(audio),
                "--language",
                "en",
                "--require-uncontended",
                "--output",
                str(output),
            ]
        )
        == exit_code
    )
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "blocked-by-contention"
    assert receipt["contention"]["status"] == status
    assert "results" not in receipt
