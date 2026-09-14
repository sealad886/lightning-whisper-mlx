"""CLI adapter for the composable benchmark runner."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, Sequence

from .runner import BenchmarkCase, BenchmarkConfig, BenchmarkSuite


def load_runner(specification: str) -> Callable:
    """Import a runner; its module must avoid model/data work at import time."""
    try:
        module_name, function_name = specification.split(":", 1)
    except ValueError as error:
        raise ValueError("runner must use module:function syntax") from error
    if not module_name or not function_name:
        raise ValueError("runner must use module:function syntax")
    runner = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(runner):
        raise ValueError(f"runner is not callable: {specification}")
    return runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        required=True,
        help="Explicit module:function callable receiving BenchmarkCase",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["none", "assistant", "mtp"], default="none"
    )
    parser.add_argument("--model")
    parser.add_argument("--quant")
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = load_runner(args.runner)
    config = BenchmarkConfig(
        cases=(
            BenchmarkCase(
                mode=args.mode,
                model=args.model,
                quant=args.quant,
                lookahead=args.lookahead,
                batch_size=args.batch_size,
            ),
        ),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    suite = BenchmarkSuite(config)
    results = suite.run(runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite.to_dict(results), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
