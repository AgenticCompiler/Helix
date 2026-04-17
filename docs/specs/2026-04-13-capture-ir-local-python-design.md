# Capture IR Local Python Design

## Summary

- Make local IR capture reuse the interpreter that launched the helper script.
- Keep remote IR capture unchanged and continue invoking `python3` on the remote host.
- Align the IR capture helper with the local interpreter behavior already used by the triton-npu-run-eval test, benchmark, and profile helpers.

## Problem

- `skills/triton-npu-analyze-ir/scripts/capture_ir.py` currently renders the benchmark command with a hard-coded `python3`.
- The same command builder is reused for both local and remote capture flows.
- This means local capture can unexpectedly escape an activated virtual environment or the code agent's current interpreter.

## Goals

- Ensure local capture uses the current interpreter.
- Preserve the existing remote contract.
- Keep the change small and easy to verify.

## Non-Goals

- Do not add a new CLI flag for Python selection.
- Do not change IR replay behavior.
- Do not change remote execution defaults.

## Design

- Update `build_execution_command()` to accept an optional `python_executable` parameter that defaults to `sys.executable`.
- Call `build_execution_command(..., python_executable="python3")` from the remote capture path.
- Leave the local capture path on the default so it inherits the current interpreter automatically.
- Update tests so local command rendering expects `sys.executable`, while the existing remote test continues to assert `python3`.

## Verification

- Run `uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v`.
