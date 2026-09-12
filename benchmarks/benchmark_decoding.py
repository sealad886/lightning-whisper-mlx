"""Reproducible decoding benchmark record format.

The runner is intentionally injected so benchmark data sources and model downloads
remain explicit and never happen during import.
"""

import argparse
import platform
import time
from pathlib import Path

import mlx


def benchmark_runner(runner, *, mode: str, repetitions: int = 3):
    durations = []
    results = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = runner()
        durations.append(time.perf_counter() - start)
        results.append(result)
    return {
        "mode": mode,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": getattr(mlx, "__version__", "unknown"),
        "repetitions": repetitions,
        "durations_seconds": durations,
        "median_seconds": sorted(durations)[len(durations) // 2],
        "exact_results": results,
        "git_revision": "record-at-invocation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["none", "assistant", "mtp"], required=True)
    parser.parse_args()
    raise SystemExit(
        "Provide a repository-specific runner to benchmark_runner; "
        "no model is downloaded by this CLI."
    )


if __name__ == "__main__":
    main()
