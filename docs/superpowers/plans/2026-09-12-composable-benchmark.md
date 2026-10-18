# Composable Benchmark Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with TDD and preserve unrelated working-tree changes.

**Goal:** Replace the placeholder decoding benchmark with a reusable, deterministic benchmark library and explicit CLI adapter.

**Architecture:** `benchmarks/runner.py` owns immutable case/config/result contracts and injected execution. The suite performs warmups, timed repetitions, normalization, and environment capture without downloading models. `benchmarks/benchmark_decoding.py` remains a thin CLI that loads an explicit runner factory and writes JSON.

**Tech Stack:** Python dataclasses, standard-library JSON/importlib/argparse/time, optional MLX version detection, pytest.

---

### Task 1: Benchmark contracts and execution

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/runner.py`
- Test: `tests/test_benchmark_runner.py`

- [ ] Write failing tests for defaults, warmup exclusion, repetition count, matrix ordering, result normalization, and invalid counts.
- [ ] Run focused tests and confirm failure because the runner API is missing.
- [ ] Implement `BenchmarkCase`, `BenchmarkConfig`, `BenchmarkResult`, `BenchmarkSuite`, and `run_benchmark` with defaults `warmups=1`, `repetitions=3`, and mode `none`.
- [ ] Capture Python/platform/MLX/git metadata without importing model loaders or downloading artifacts.
- [ ] Run focused tests and confirm green.

### Task 2: Explicit CLI runner loading

**Files:**
- Modify: `benchmarks/benchmark_decoding.py`
- Test: `tests/test_benchmark_cli.py`

- [ ] Write failing tests for explicit `module:function` runner loading, JSON output, and missing-runner failure.
- [ ] Run focused tests and confirm failure against the placeholder CLI.
- [ ] Implement CLI arguments for `--runner`, `--output`, `--mode`, `--warmups`, and `--repetitions`; write serialized suite results.
- [ ] Run focused tests and confirm green.

### Task 3: Documentation and full verification

**Files:**
- Modify: `docs/evaluation/decoding-benchmarks.md`

- [ ] Document Python API, defaults, runner callable contract, CLI usage, and no-implicit-download behavior.
- [ ] Run the complete test suite, compileall, and focused lint.
- [ ] Inspect diff and confirm user-edited files remain untouched.
