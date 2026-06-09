# Remote Msprof Runtime Staging Regression Design

## Summary

Fix remote serial `run-bench --bench-mode msprof` so it stages the shared benchmark runtime support files before invoking `bench_runtime.py run-one`, matching the existing behavior already used by remote standalone and remote parallel `msprof`.

## Goals

- Make remote serial `msprof` execution work with the unified import-only benchmark contract.
- Keep `standalone` and `msprof` benchmark case code identical; only profiling behavior should differ at execution time.
- Add regression coverage that proves remote serial `msprof` stages the required runtime support files.

## Non-Goals

- Do not redesign benchmark case loading or remote workspace layout.
- Do not change local execution behavior.
- Do not introduce a new remote execution abstraction layer.

## Decision

- Before remote serial `msprof` begins running cases, copy every path returned by `_bench_runtime_support_paths()` into the remote workspace root.
- Keep the existing per-case `msprof --output=... python3 bench_runtime.py run-one ...` execution contract unchanged.
- Reuse the same runtime support staging contract already exercised by the remote standalone path instead of adding a new mode-specific mechanism.

## Verification

- Add a focused remote execution test that fails if remote serial `msprof` does not copy `bench_runtime.py` and the other runtime support files into the remote workspace.
- Run the focused remote execution test file.
- Run the strict file-scoped pyright check for `skills/triton-npu-run-eval/scripts/bench_runner.py`.
