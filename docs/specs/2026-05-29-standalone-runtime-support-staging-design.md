# Standalone Runtime Support Staging Design

## Summary

Parallel `run-bench --bench-mode standalone` workers and remote standalone helpers copy
`standalone_bench_runtime.py` into isolated workspaces. That runtime now imports
`profile_csv_parser.py`, but the staged support-file list was not updated, so isolated
workers fail with `ModuleNotFoundError`.

## Goals

- Restore local parallel standalone benchmark execution.
- Keep remote standalone benchmark and profile flows staging the full runtime dependency set.
- Remove the separate, incomplete support-file list in IR capture so the runtime contract has one source of truth.

## Non-Goals

- Do not change standalone benchmark semantics or CLI flags.
- Do not refactor unrelated profiling or perf parsing behavior.

## Decision

- Extend `standalone_bench_runtime.runtime_support_paths()` to include `profile_csv_parser.py`.
- Reuse that helper from IR capture instead of maintaining a shorter hard-coded list there.
- Add regression tests that assert the staged support-file set includes the CSV parser helper.

## Verification

- Run focused unit tests for bench-runner and IR-capture support staging.
- Run the file-scoped strict `pyright` check for
  `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`.
