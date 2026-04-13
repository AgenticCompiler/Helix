# Profile Bench Local Python Design

## Summary

- Make local `profile-bench` execution reuse the interpreter that launched `run-command.py`.
- Keep remote `profile-bench` execution unchanged and still invoke `python3` on the remote host.
- Align local `profile-bench` behavior with existing local `run-test` and `run-bench` behavior.

## Problem

- Local `profile-bench` currently hard-codes `python3` in its benchmark and case-count commands.
- This breaks consistency with the rest of the local operator-eval helpers, which already reuse the current interpreter via `sys.executable`.
- When a user or code agent launches the helper from an activated virtual environment, local profiling may escape that environment unexpectedly.

## Goals

- Ensure local profiling honors the current Python interpreter.
- Preserve existing CLI arguments and output behavior.
- Minimize code changes and regression risk.

## Non-Goals

- Do not change remote profiling behavior.
- Do not introduce a new CLI flag for selecting Python.
- Do not refactor unrelated operator-eval helpers.

## Design

- Import `sys` in `skills/operator-eval/scripts/profile_runner.py`.
- Replace local hard-coded `python3` command entries with `sys.executable` in:
  - standalone local profile execution
  - msprof local profile execution
  - local benchmark case-count probing
- Update unit tests to assert the local commands now use `sys.executable`.

## Verification

- Run `uv run python -m unittest tests.test_profile_runner -v`.
