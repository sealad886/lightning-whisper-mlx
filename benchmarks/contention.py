"""Read-only, bounded macOS contention observations; no MLX or process mutations.

IORegistry utilization and per-client Metal counters are observational proxies.
Passing means low contention was observed within the declared windows, never that
all activity between samples or all shared-resource interference was excluded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ctypes
import math
import os
import plistlib
import re
import subprocess
import time


@dataclass(frozen=True)
class ContentionPolicy:
    samples: int = 3
    interval_seconds: float = 0.5
    cooldown_seconds: float = 2.0
    max_gpu_percent: float = 5.0
    max_cpu_busy_percent: float = 20.0
    max_compute_cpu_percent: float = 5.0
    min_memory_free_percent: float = 10.0
    max_thermal_state: int = 0
    require_ac: bool = True
    allowed_gpu_names: tuple[str, ...] = ("WindowServer",)

    def __post_init__(self):
        if not 3 <= self.samples <= 10:
            raise ValueError("contention samples must be between 3 and 10")
        if (
            not 0.1 <= self.interval_seconds <= 5
            or not 0 <= self.cooldown_seconds <= 10
        ):
            raise ValueError("contention interval/cooldown outside bounded range")
        for value in (
            self.max_gpu_percent,
            self.max_cpu_busy_percent,
            self.min_memory_free_percent,
        ):
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(
                    "contention percentages must be finite and between 0 and 100"
                )
        if (
            not math.isfinite(self.max_compute_cpu_percent)
            or self.max_compute_cpu_percent < 0
        ):
            raise ValueError("compute CPU threshold must be finite and nonnegative")
        if self.max_thermal_state not in range(4):
            raise ValueError(
                "thermal state must be 0 nominal, 1 fair, 2 serious, or 3 critical"
            )


def _cpu_ticks():
    # Public Darwin HOST_CPU_LOAD_INFO: user/system/idle/nice uint32 tick counts.
    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    system.mach_host_self.restype = ctypes.c_uint
    system.host_statistics.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    info, count = (ctypes.c_uint * 4)(), ctypes.c_uint(4)
    if (
        system.host_statistics(system.mach_host_self(), 3, info, ctypes.byref(count))
        or count.value != 4
    ):
        raise RuntimeError("Darwin CPU load counters unavailable")
    return list(info)


def collect_snapshot():
    """Collect small filtered snapshots, recording unavailable telemetry explicitly."""
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "errors": {}}

    def read(name, command, *, binary=False):
        try:
            return subprocess.check_output(
                command,
                text=not binary,
                stderr=subprocess.PIPE,
                timeout=3,
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            result["errors"][name] = type(error).__name__ + ": " + str(error)[:250]
            return None

    try:
        result["cpu_ticks"] = _cpu_ticks()
    except (OSError, AttributeError, RuntimeError) as error:
        result["cpu_ticks"] = None
        result["errors"]["cpu"] = str(error)
    raw = read("gpu", ["ioreg", "-ar", "-c", "IOAccelerator", "-d", "1"], binary=True)
    result["gpu_device_percent"] = None
    if raw is not None:
        try:
            nodes = plistlib.loads(raw)
            values = [
                node["PerformanceStatistics"]["Device Utilization %"]
                for node in nodes
                if "Device Utilization %" in node.get("PerformanceStatistics", {})
            ]
            result["gpu_device_percent"] = max(values) if values else None
        except (
            ValueError,
            TypeError,
            KeyError,
            plistlib.InvalidFileException,
        ) as error:
            result["errors"]["gpu"] = str(error)
    raw = read(
        "gpu_clients",
        ["ioreg", "-ar", "-c", "AGXDeviceUserClient", "-d", "1"],
        binary=True,
    )
    result["gpu_clients"] = None
    if raw is not None:
        try:
            clients = {}
            nodes = plistlib.loads(raw)
            for node in nodes:
                creator = node.get("IOUserClientCreator", "")
                match = re.fullmatch(r"pid (\d+), (.+)", creator)
                if not match:
                    continue
                pid, name = int(match[1]), match[2]
                for usage in node.get("AppUsage", []):
                    if usage.get("API") != "Metal":
                        continue
                    client = clients.setdefault(
                        (pid, name),
                        {
                            "pid": pid,
                            "name": name,
                            "gpu_time_raw": 0,
                            "last_submission_raw": 0,
                        },
                    )
                    client["gpu_time_raw"] += usage.get("accumulatedGPUTime", 0)
                    client["last_submission_raw"] = max(
                        client["last_submission_raw"], usage.get("lastSubmittedTime", 0)
                    )
            result["gpu_clients"] = (
                list(clients.values()) if nodes and clients else None
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            plistlib.InvalidFileException,
        ) as error:
            result["errors"]["gpu_clients"] = str(error)
    raw = read(
        "thermal",
        [
            "/usr/bin/osascript",
            "-l",
            "JavaScript",
            "-e",
            'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState',
        ],
    )
    result["thermal_state"] = (
        int(raw.strip())
        if raw is not None and raw.strip() in {"0", "1", "2", "3"}
        else None
    )
    raw = read("power", ["pmset", "-g", "batt"])
    match = re.search(r"Now drawing from '([^']+)'", raw or "")
    result["power_source"] = match[1] if match else None
    raw = read("memory", ["memory_pressure", "-Q"])
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", raw or "")
    result["memory_free_percent"] = float(match[1]) if match else None
    raw = read("vm", ["vm_stat"])
    result["vm_counters"] = {}
    for name in ("Pageouts", "Swapins", "Swapouts"):
        match = re.search(r"^" + name + r":\s*(\d+)", raw or "", re.MULTILINE)
        result["vm_counters"][name.lower()] = int(match[1]) if match else None
    raw = read("processes", ["ps", "-axo", "pid=,ppid=,pcpu=,comm="])
    processes = []
    for line in (raw or "").splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4:
            try:
                pid, parent, cpu, comm = fields
                processes.append(
                    dict(
                        pid=int(pid),
                        ppid=int(parent),
                        cpu_percent=float(cpu),
                        comm=comm,
                    )
                )
            except ValueError:
                continue
    # Never collect arbitrary command arguments, environment variables or prompts.
    result["processes"] = (
        sorted(processes, key=lambda process: process["cpu_percent"], reverse=True)[:20]
        if raw is not None
        else None
    )
    return result


def _gpu_changes(previous, current, policy, owned_pids):
    activity, unknown = [], []
    if previous.get("gpu_clients") is None or current.get("gpu_clients") is None:
        return activity, ["per-client GPU interval unavailable"]
    before = {
        (client["pid"], client["name"]): client for client in previous["gpu_clients"]
    }
    for client in current["gpu_clients"]:
        if client["pid"] in owned_pids:
            continue
        old = before.get((client["pid"], client["name"]))
        delta = (
            client["gpu_time_raw"] - old["gpu_time_raw"] if old is not None else None
        )
        active = (
            (client["gpu_time_raw"] > 0 or client["last_submission_raw"] > 0)
            if old is None
            else (
                delta > 0 or client["last_submission_raw"] > old["last_submission_raw"]
            )
        )
        if delta is not None and delta < 0:
            unknown.append(f"GPU client counter reset for pid {client['pid']}")
        if (
            active
            and client["pid"] not in owned_pids
            and client["name"] not in policy.allowed_gpu_names
        ):
            activity.append({**client, "gpu_time_raw_delta": delta})
    return activity, unknown


def _paging_changes(previous, current):
    reasons, unknown = [], []
    for counter in ("pageouts", "swapins", "swapouts"):
        start = previous.get("vm_counters", {}).get(counter)
        end = current.get("vm_counters", {}).get(counter)
        if start is None or end is None or end < start:
            unknown.append(f"{counter} interval unavailable")
        elif end > start:
            reasons.append(f"active paging {counter} delta {end-start}")
    return reasons, unknown


def assess_window(observations, policy=ContentionPolicy(), *, owned_pids=()):
    """Distinguish missing evidence from observed competition; never infer GPU from CPU."""
    reasons, unknown, gpu_activity, cpu_busy = [], [], [], []
    owned_pids = set(owned_pids)
    if len(observations) < policy.samples:
        unknown.append("insufficient samples")
    for index, sample in enumerate(observations):
        for field in (
            "cpu_ticks",
            "gpu_device_percent",
            "gpu_clients",
            "thermal_state",
            "power_source",
            "memory_free_percent",
            "processes",
        ):
            if sample.get(field) is None:
                unknown.append(f"sample {index}: {field} unavailable")
        if sample.get("errors"):
            unknown.extend(
                f"sample {index}: {name} telemetry failed" for name in sample["errors"]
            )
        gpu = sample.get("gpu_device_percent")
        if gpu is not None and gpu > policy.max_gpu_percent:
            reasons.append(
                f"sample {index}: GPU {gpu}% exceeds {policy.max_gpu_percent}% WindowServer/background budget"
            )
        thermal = sample.get("thermal_state")
        if thermal is not None and thermal > policy.max_thermal_state:
            reasons.append(
                f"sample {index}: thermal state {thermal} exceeds {policy.max_thermal_state}"
            )
        if policy.require_ac and sample.get("power_source") not in (None, "AC Power"):
            reasons.append(f"sample {index}: AC power required")
        memory = sample.get("memory_free_percent")
        if memory is not None and memory < policy.min_memory_free_percent:
            reasons.append(
                f"sample {index}: memory headroom {memory}% below {policy.min_memory_free_percent}%"
            )
        for process in sample.get("processes") or []:
            if process["pid"] in owned_pids or process.get("ppid") in owned_pids:
                continue
            if process["cpu_percent"] > policy.max_compute_cpu_percent and re.search(
                r"python|graphify|mlx|whisper|ollama|llama|rustc|clang|xcodebuild|cargo|(^|/)node$",
                process["comm"],
                re.I,
            ):
                reasons.append(
                    f"sample {index}: competing compute pid {process['pid']} {process['comm']} uses {process['cpu_percent']}% CPU"
                )
        if not index:
            continue
        previous = observations[index - 1]
        left, right = previous.get("cpu_ticks"), sample.get("cpu_ticks")
        if left is not None and right is not None:
            deltas = [end - start for start, end in zip(left, right)]
            if len(deltas) != 4 or min(deltas) < 0 or sum(deltas) <= 0:
                unknown.append(f"sample {index}: invalid CPU counter interval")
            else:
                busy = 100 * (sum(deltas) - deltas[2]) / sum(deltas)
                cpu_busy.append(busy)
                if busy > policy.max_cpu_busy_percent:
                    reasons.append(
                        f"sample {index}: total CPU busy {busy:.2f}% exceeds {policy.max_cpu_busy_percent}%"
                    )
        paging, invalid = _paging_changes(previous, sample)
        reasons.extend(f"sample {index}: {reason}" for reason in paging)
        unknown.extend(f"sample {index}: {reason}" for reason in invalid)
        activity, invalid = _gpu_changes(previous, sample, policy, owned_pids)
        unknown.extend(f"sample {index}: {reason}" for reason in invalid)
        for client in activity:
            gpu_activity.append({"sample_index": index, **client})
            reasons.append(
                f"sample {index}: competing Metal GPU work pid {client['pid']} {client['name']}"
            )

    return {
        "status": (
            "unknown" if unknown else "busy" if reasons else "observed-low-contention"
        ),
        "reasons": reasons,
        "unknown": unknown,
        "external_gpu_activity": gpu_activity,
        "cpu_busy_percent": cpu_busy,
        "policy": asdict(policy),
        "observations": observations,
        "guarantees_uncontended": False,
        "limits": "sampled IORegistry GPU proxies and OS counters; activity between windows and all memory-bandwidth interference cannot be excluded; raw client time units are not treated as utilization percentages",
    }


class ContentionError(RuntimeError):
    pass


class QuietGuard:
    def __init__(
        self,
        policy=ContentionPolicy(),
        *,
        required=False,
        owned_pids=(),
        collector=collect_snapshot,
        sleeper=time.sleep,
    ):
        self.policy, self.required = policy, required
        self.owned_pids = {os.getpid(), *owned_pids}
        self.collector, self.sleeper = collector, sleeper
        self.checks = []

    def check(self, phase, *, cooldown=False):
        if cooldown and self.policy.cooldown_seconds:
            self.sleeper(self.policy.cooldown_seconds)
        observations = []
        for index in range(self.policy.samples):
            if index:
                self.sleeper(self.policy.interval_seconds)
            observations.append(self.collector())
        report = {
            "phase": phase,
            **assess_window(observations, self.policy, owned_pids=self.owned_pids),
        }
        # Compare client counters across the timed gap as well. Do not compare
        # total CPU/device utilization there: the benchmark itself owns that work.
        if self.checks:
            activity, invalid = _gpu_changes(
                self.checks[-1]["observations"][-1],
                observations[0],
                self.policy,
                self.owned_pids,
            )
            paging, invalid_paging = _paging_changes(
                self.checks[-1]["observations"][-1], observations[0]
            )
            report["reasons"].extend("between checks: " + reason for reason in paging)
            report["unknown"].extend(
                "between checks: " + reason for reason in invalid_paging
            )
            report["between_checks_gpu_activity"] = activity
            report["unknown"].extend("between checks: " + reason for reason in invalid)
            report["reasons"].extend(
                f"between checks: competing Metal GPU work pid {client['pid']} {client['name']}"
                for client in activity
            )
            report["status"] = (
                "unknown"
                if report["unknown"]
                else "busy" if report["reasons"] else "observed-low-contention"
            )
        self.checks.append(report)
        if self.required and report["status"] != "observed-low-contention":
            raise ContentionError(
                f"{phase}: contention evidence {report['status']}; see recorded observations"
            )
        return report

    def summary(self):
        statuses = [check["status"] for check in self.checks]
        status = (
            "unknown"
            if not statuses or "unknown" in statuses
            else "busy" if "busy" in statuses else "observed-low-contention"
        )
        return {
            "required": self.required,
            "status": status,
            "qualified": self.required and status == "observed-low-contention",
            "guarantees_uncontended": False,
            "checks": self.checks,
            "scope": "bounded quiet windows before/after paired rounds; sampled evidence, not guaranteed exclusive hardware ownership",
        }
