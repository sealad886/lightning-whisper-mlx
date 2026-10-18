"""Composable, side-effect-free benchmark execution."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
import subprocess
import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark configuration passed unchanged to the injected runner."""

    mode: str = "none"
    model: Optional[str] = None
    quant: Optional[str] = None
    lookahead: int = 2
    batch_size: int = 1
    label: Optional[str] = None

    def __post_init__(self):
        if self.mode not in {"none", "assistant", "mtp"}:
            raise ValueError("mode must be one of: none, assistant, mtp")
        if self.lookahead < 1:
            raise ValueError("lookahead must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Execution policy; defaults avoid network work and noisy single samples."""

    cases: tuple[BenchmarkCase, ...] = field(default_factory=lambda: (BenchmarkCase(),))
    warmups: int = 1
    repetitions: int = 3

    def __post_init__(self):
        if self.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not self.cases:
            raise ValueError("cases must not be empty")


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    durations_seconds: tuple[float, ...]
    repetitions: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    metadata: Mapping[str, Any]
    output: Any = None
    token_count: Optional[int] = None
    outputs: tuple[Any, ...] = ()
    sample_order: tuple[Mapping[str, int], ...] = ()
    sample_metrics: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["case"] = asdict(self.case)
        value["durations_seconds"] = list(self.durations_seconds)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def environment_metadata() -> dict[str, Any]:
    """Return stable local metadata without model loading or network access."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        dirty = None

    try:
        mlx_version = version("mlx")
    except PackageNotFoundError:
        mlx_version = "unavailable"

    import hashlib

    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    paths = [root / "setup.py", root / "pyproject.toml"]
    for directory in ("lightning_whisper_mlx", "benchmarks", "training"):
        paths.extend((root / directory).rglob("*.py"))
    for path in sorted(paths):
        if path.is_file():
            source.update(str(path.relative_to(root)).encode() + b"\0")
            source.update(path.read_bytes())
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": mlx_version,
        "source_tree_sha256": source.hexdigest(),
    }


def _normalize_output(output: Any) -> tuple[Any, Optional[int]]:
    if isinstance(output, Mapping):
        token_count = output.get("token_count")
        if token_count is None and "tokens" in output:
            try:
                token_count = len(output["tokens"])
            except TypeError:
                token_count = None
        return output, token_count
    return output, None


def run_benchmark(
    runner: Callable[[BenchmarkCase], Any],
    case: BenchmarkCase,
    *,
    warmups: int = 1,
    repetitions: int = 3,
    metadata: Optional[Mapping[str, Any]] = None,
) -> BenchmarkResult:
    """Run one case; warmups execute but never enter timing statistics."""
    config = BenchmarkConfig(cases=(case,), warmups=warmups, repetitions=repetitions)
    return BenchmarkSuite(config).run(runner, metadata=metadata)[0]


class BenchmarkSuite:
    def __init__(self, config: BenchmarkConfig = BenchmarkConfig()):
        self.config = config
        self.execution_order: list[dict[str, Any]] = []

    def run(
        self,
        runner: Callable[[BenchmarkCase], Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        before_sample: Optional[Callable[[], None]] = None,
        after_sample: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> list[BenchmarkResult]:
        """Interleave cases, rotating their starting position each round.

        Optional resource measurement hooks run outside the timed interval.
        Runners must synchronize their asynchronous work before returning.
        """
        self.execution_order = []
        provenance = {**(metadata or {}), **environment_metadata()}
        count = len(self.config.cases)
        durations = [[] for _ in self.config.cases]
        outputs = [[] for _ in self.config.cases]
        token_counts = [None for _ in self.config.cases]
        sample_orders = [[] for _ in self.config.cases]
        metrics = [[] for _ in self.config.cases]
        for phase, rounds in (
            ("warmup", self.config.warmups),
            ("timed", self.config.repetitions),
        ):
            for round_index in range(rounds):
                for position in range(count):
                    case_index = (round_index + position) % count
                    case = self.config.cases[case_index]
                    order = {
                        "round_index": round_index,
                        "position": position,
                        "case_index": case_index,
                        "execution_index": len(self.execution_order),
                    }
                    self.execution_order.append({"phase": phase, **order})
                    if phase == "warmup":
                        runner(case)
                        continue
                    if before_sample is not None:
                        before_sample()
                    start = time.perf_counter()
                    value = runner(case)
                    duration = time.perf_counter() - start
                    measured = after_sample() if after_sample is not None else {}
                    normalized, token_count = _normalize_output(value)
                    durations[case_index].append(duration)
                    outputs[case_index].append(deepcopy(normalized))
                    token_counts[case_index] = token_count
                    sample_orders[case_index].append(order)
                    metrics[case_index].append(deepcopy(measured))

        return [
            BenchmarkResult(
                case=case,
                durations_seconds=tuple(durations[index]),
                repetitions=self.config.repetitions,
                median_seconds=median(durations[index]),
                min_seconds=min(durations[index]),
                max_seconds=max(durations[index]),
                metadata=provenance,
                output=deepcopy(outputs[index][-1]),
                token_count=token_counts[index],
                outputs=tuple(outputs[index]),
                sample_order=tuple(sample_orders[index]),
                sample_metrics=tuple(metrics[index]),
            )
            for index, case in enumerate(self.config.cases)
        ]

    def to_dict(self, results: Sequence[BenchmarkResult]) -> dict[str, Any]:
        return {
            "config": {
                "warmups": self.config.warmups,
                "repetitions": self.config.repetitions,
                "execution_policy": "interleaved-rotating-start",
            },
            "execution_order": deepcopy(self.execution_order),
            "results": [result.to_dict() for result in results],
        }
