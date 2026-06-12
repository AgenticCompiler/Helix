# Run-Command Blocks-Parallel Env Guard Design

## Summary

- Guard `TRITON_ALL_BLOCKS_PARALLEL` inside the `triton-npu-run-eval` execution entrypoint so operator execution does not inherit the unsafe value `1`.
- Keep the behavior scoped to run-eval operator execution commands instead of changing unrelated CLI or compare-only flows.
- Preserve remote execution parity by forwarding the guarded value through the shared remote command environment assembly layer.

## Problem

- Optimize-time agents may leave `TRITON_ALL_BLOCKS_PARALLEL=1` in the environment.
- Subsequent operator execution can then inherit that value and trigger correctness or precision issues.
- The run-eval stack has both local in-process execution and remote command execution, so fixing only one layer can leave gaps.

## Goals

- Before `run-test`, `run-test-baseline`, `run-test-optimize`, `run-bench`, or `profile-bench` execute operator code, detect `TRITON_ALL_BLOCKS_PARALLEL=1` and force it to `0`.
- Restore the caller environment after the guarded command finishes.
- Ensure remote run-eval commands explicitly receive `TRITON_ALL_BLOCKS_PARALLEL=0` when the local run-command entrypoint has activated the guard.

## Non-Goals

- Do not change compare-only commands such as `compare-result`, `compare-perf`, or `profile-report`.
- Do not redesign the run-eval runner boundaries or introduce a new global runtime policy outside the skill-local execution path.

## Decision

- Add a small guard context in `skills/triton-npu-run-eval/scripts/run-command.py`.
- Activate that guard only for run-eval commands that execute operator code:
  - `run-test`
  - `run-test-baseline`
  - `run-test-optimize`
  - `run-bench`
  - `profile-bench`
- When the guard sees `TRITON_ALL_BLOCKS_PARALLEL=1`, temporarily rewrite it to `0` for the duration of the command and restore the original value afterward.
- Update the shared remote environment prefix logic in `skills/triton-npu-run-eval/scripts/run_runtime.py` so a guarded local environment also becomes an explicit remote environment assignment.

## Verification

- Add a run-command test that proves local operator execution sees `TRITON_ALL_BLOCKS_PARALLEL=0` while the entrypoint is active and that the original environment value is restored afterward.
- Add a remote execution test that proves the shared remote command wrapper prefixes `TRITON_ALL_BLOCKS_PARALLEL=0` when the guarded environment is active.
- Run focused unit tests for the changed behavior.
- Run the strict skill-script pyright wrapper for every modified file under `skills/*/scripts/`.
