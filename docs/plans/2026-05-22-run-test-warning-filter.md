# Run-Test Warning Filter Implementation Plan

## Summary

Suppress `run-test` output lines that begin with `[WARNING]` during non-verbose execution while preserving full output in `--verbose` mode. Spec: `docs/specs/2026-05-22-run-test-warning-filter-design.md`.

## Files Changed

### Core behavior: `skills/triton-npu-run-eval/scripts/test_runner.py`

1. Add a small helper that removes result-payload lines whose rendered text starts with `[WARNING]`.
2. Thread `verbose` into local `run_local_test()` and legacy local execution so non-verbose runs can filter result output.
3. Apply the same filtering to remote `run-test` results before returning them.

### Repo CLI wrapper: `src/triton_agent/execution.py`

1. Extend `run_local_test()` to accept `verbose`.
2. Update the protocol definition to match the new test-runner signature.

### Repo CLI command: `src/triton_agent/commands/execution.py`

1. Pass `args.verbose` into local `run_local_test()`.

### Skill helper CLI: `skills/triton-npu-run-eval/scripts/run-command.py`

1. Extend the local test protocol to accept `verbose`.
2. Pass `args.verbose` into local `run-test`.

### Tests

- `tests/test_test_runner.py`
- `tests/test_execution_commands.py`
- `tests/test_skill_command_script.py`

## Implementation Order

1. Add failing tests for filtered default local behavior, verbose passthrough, and filtered remote behavior in `tests/test_test_runner.py`.
2. Add failing wrapper tests for repository CLI and skill helper `run-command.py` verbose propagation.
3. Implement the minimal filtering helper and thread `verbose` through local and remote `run-test` paths.
4. Run focused unit tests plus `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/test_runner.py`.
