# Bug Review Status — 2026-04-02

This document reflects the current status after fixing the confirmed regressions from the 2026-04-02 review.

## Resolved In Code

### 1. Stall timeout default trap
**Status:** Fixed

`src/triton_agent/process_runner.py` now treats `stall_timeout_seconds <= 0` as "stall timeout disabled" instead of immediately reporting a stall on the first idle poll.

### 2. `returncode or 0` fallback
**Status:** Fixed defensively

Both `src/triton_agent/process_runner.py` and `skills/run-validation/scripts/run_runtime.py` now turn a missing `returncode` into `1` instead of silently reporting success.

### 3. Supervisor re-ran after repeated stalls
**Status:** Fixed

`src/triton_agent/supervisor.py` now stays on the resume path after repeated stalls instead of falling back to a fresh `run()` call.

### 4. Diff filter swallowed normal indented output
**Status:** Fixed

`src/triton_agent/codex_runner.py` now keeps normal indented output that appears after a diff block instead of treating every space-prefixed line as diff content forever.

### 5. Remote command quoting
**Status:** Fixed

`skills/run-validation/scripts/run_runtime.py`, `skills/run-validation/scripts/test_runner.py`, and `skills/run-validation/scripts/bench_runner.py` now shell-join remote command arguments so filenames with spaces and shell metacharacters are quoted correctly.

### 6. `_normalize_agent_result` raw `KeyError`
**Status:** Fixed

`src/triton_agent/cli.py` now raises a short `ValueError` that explains which required keys are missing from a run-skill payload.

## Re-Triaged Items

### 7. Broken remote comparison script path
**Status:** Not reproduced

`skills/run-validation/scripts/test_runner.py` resolves `compare_result_payloads.py` to `<repo>/scripts/compare_result_payloads.py`, and that file exists in the repository.

### 8. Unbounded `lru_cache` on run-skill loading
**Status:** Still a design tradeoff

`src/triton_agent/run_skill.py` intentionally caches dynamically loaded run-skill modules for the lifetime of the current CLI process. That behavior may be worth documenting more explicitly, but it is not a newly confirmed functional regression.

### 9. `remote_workspace` local variable scoping
**Status:** Low-risk code smell only

`src/triton_agent/cli.py` only reads `remote_workspace` in branches where the remote execution path has already assigned it. Initializing it to `None` would be harmless, but current control flow does not expose an `UnboundLocalError`.

### 10. Missing `agents` color category
**Status:** Cosmetic only

`src/triton_agent/verbose.py` still renders the `agents` prefix without a dedicated color entry. Output remains readable and functional.

## Verification

The following regression coverage was added with this update:

- `tests/test_process_runner.py`
- `tests/test_supervisor.py`
- `tests/test_remote_execution.py`
- `tests/test_cli.py`
