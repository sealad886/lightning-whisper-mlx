# Composable Benchmark Design

**Status:** Approved

## Goal

Provide reusable benchmark execution for decoding modes without implicit model
loading, downloads, or hidden data selection.

## Design

`BenchmarkCase` describes one model/mode configuration. `BenchmarkConfig` owns
safe execution defaults: one warmup and three timed repetitions, with one `none`
case by default. `BenchmarkSuite` runs cases in declared order using an injected
callable. `BenchmarkResult` records per-run timings, summary statistics, runner
output, token count when available, and local environment metadata.

The CLI is a thin adapter. It requires an explicit `module:function` runner,
constructs one case, and writes JSON. This keeps model loading and audio fixtures
outside the benchmark framework and makes notebook, CI, and command-line callers
share the same execution contract.

## Verification

Tests cover default configuration, warmup exclusion, repetition counts, matrix
ordering, output normalization, invalid counts, explicit runner loading, JSON
output, and missing runner arguments.
