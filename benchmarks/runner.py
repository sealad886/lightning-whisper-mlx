"""Composable, side-effect-free benchmark execution."""

from __future__ import annotations

import json
import platform
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

    cases: tuple[BenchmarkCase, ...] = field(
        default_factory=lambda: (BenchmarkCase(),)
    )
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

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["case"] = asdict(self.case)
        value["durations_seconds"] = list(self.durations_seconds)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def environment_metadata() -> dict[str, str]:
    """Return stable local metadata without model loading or network access."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"

    try:
        import mlx

        mlx_version = getattr(mlx, "__version__", "unknown")
    except ImportError:
        mlx_version = "unavailable"

    return {
        "git_revision": revision,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": mlx_version,
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
    if warmups < 0:
        raise ValueError("warmups must be nonnegative")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    for _ in range(warmups):
        runner(case)

    durations = []
    output = None
    token_count = None
    for _ in range(repetitions):
        start = time.perf_counter()
        output, token_count = _normalize_output(runner(case))
        durations.append(time.perf_counter() - start)

    return BenchmarkResult(
        case=case,
        durations_seconds=tuple(durations),
        repetitions=repetitions,
        median_seconds=median(durations),
        min_seconds=min(durations),
        max_seconds=max(durations),
        metadata={**environment_metadata(), **(metadata or {})},
        output=output,
        token_count=token_count,
    )


class BenchmarkSuite:
    def __init__(self, config: BenchmarkConfig = BenchmarkConfig()):
        self.config = config

    def run(
        self,
        runner: Callable[[BenchmarkCase], Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> list[BenchmarkResult]:
        return [
            run_benchmark(
                runner,
                case,
                warmups=self.config.warmups,
                repetitions=self.config.repetitions,
                metadata=metadata,
            )
            for case in self.config.cases
        ]

    def to_dict(self, results: Sequence[BenchmarkResult]) -> dict[str, Any]:
        return {
            "config": {
                "warmups": self.config.warmups,
                "repetitions": self.config.repetitions,
            },
            "results": [result.to_dict() for result in results],
        }
