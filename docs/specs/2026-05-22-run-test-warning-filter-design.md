# Run-Test Warning Filter Design

## Goal

Hide `run-test` warning lines that begin with `[WARNING]` unless the user explicitly opts into verbose output.

## Problem

`run-test` currently forwards legacy script-mode output directly to the terminal. Some Ascend Triton operators emit repeated warning lines such as `[WARNING] Please DO NOT tune args ['num_warps']!` during otherwise successful correctness runs. These lines are noisy, not actionable for normal validation, and can appear once per test case.

## User-Visible Behavior

- `run-test` should suppress any whole output line whose rendered text starts with `[WARNING]` by default for both local and remote execution.
- `run-test --verbose` should preserve the warning output exactly as emitted by the underlying test process.
- Other stdout and stderr output should remain unchanged.
- `run-bench` and unrelated commands should keep their current behavior.

## Design

Keep the behavior change inside the `run-test` execution path.

- Add a small filtering helper in `skills/triton-npu-run-eval/scripts/test_runner.py` that removes output lines whose rendered text starts with `[WARNING]` from `stdout` and `stderr`.
- Thread a `verbose` flag through local `run-test` entry points so the helper can skip filtering when verbose mode is enabled.
- Apply the same filtering to remote `run-test` results after the streaming command completes so local and remote paths stay aligned.
- Update both the repository CLI wrapper and the skill-side `run-command.py` helper to pass the local `--verbose` flag into the test runner.

## Testing

- Add coverage that local legacy `run-test` filters the warning by default.
- Add coverage that local legacy `run-test --verbose` preserves the warning.
- Add coverage that remote legacy `run-test` filters the warning by default.
- Add wrapper tests that confirm CLI layers pass `verbose=True` to local `run-test` when requested.

## Scope

- Do not add generic warning filtering for all commands.
- Do not change declarative differential test execution beyond sharing the filtered result helper.
- Do not apply this warning filtering outside `run-test`.
