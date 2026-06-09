# Bench Runner Single File Simplification Design

## Summary

Simplify benchmark execution orchestration without changing user-facing benchmark mode semantics:

- `bench-mode` remains the switch that chooses standalone versus `msprof` execution behavior.
- `skills/triton-npu-run-eval/scripts/bench_runner.py` becomes the single concrete runner implementation.
- Delete the redundant dependency adapter and split mode wrapper modules that only forwarded to other functions.

## Goals

- Keep `run-bench` and profile-related behavior unchanged for local and remote execution.
- Make the bench runner easier to read by removing fake dependency injection and multi-file forwarding.
- Keep shared payload construction and shared runtime helpers sourced from their real owning modules.

## Non-Goals

- Do not remove `bench-mode` or collapse standalone and `msprof` into one execution path.
- Do not redesign benchmark case generation or `bench_runtime.py`.
- Do not add compatibility shims for deleted internal modules.

## Decision

### Keep one concrete bench runner

- `bench_runner.py` owns benchmark orchestration directly.
- Keep existing entrypoints and helper names when they still describe real behavior and help tests stay focused.
- Delete `bench_runner_deps.py`, `bench_runner_msprof.py`, and `bench_runner_standalone.py`.

### Remove fake abstraction layers

- Delete `BenchRunnerDeps` and any service-locator style indirection.
- Import shared types and helpers from their real source modules instead of bouncing through wrapper layers.
- Prefer direct local function calls over mode-specific forwarding files.

### Preserve visible semantics

- Local and remote standalone execution continue to use standalone runtime behavior.
- Local and remote `msprof` execution continue to use profiler-specific collection behavior.
- Failures, perf output generation, and artifact handling keep their existing contracts unless a separate change explicitly updates them.

## Verification

- Add source-structure coverage that asserts the deleted dependency adapter and mode wrapper imports are gone.
- Add loader coverage that asserts the deleted files no longer exist in the skill script directory.
- Run focused benchmark and remote execution tests after the refactor.
