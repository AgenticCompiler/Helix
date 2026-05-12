# Optimize/Vverify PT File Auto-Cleanup

**Date**: 2026-05-12

## Motivation

During `triton-agent optimize`, every correctness test run in differential mode produces a `*_result.pt` file — a binary archive containing serialized tensors from all test cases. These files can be large (hundreds of MB) and accumulate at multiple locations:

- Workspace root: from testing the original or intermediate operator files
- `baseline/`: from baseline correctness validation
- `opt-round-N/`: from per-round correctness validation
- `opt-verify/verify-*/`: from verification re-runs

Once correctness is confirmed and recorded in `round-state.json` / `baseline/state.json` / `verify-state.json`, these binary archives are redundant and waste disk space.

## Design

### Shared Cleanup Module

A single shared module `src/triton_agent/optimize/pt_cleanup.py` provides two functions:

- `cleanup_dir_pt_files(directory: Path) -> list[str]` — cleans `*_result.pt` and `test_result.pt` (case-insensitive) from a single directory
- `cleanup_workspace_pt_files(workdir: Path) -> list[str]` — cleans root-level pt files plus all `opt-round-*/` subdirectories

All cleanup is guarded by `try/except OSError` so failures (permissions, disk full) never propagate upstream.

### Trigger Points

| Trigger | Location | Scope |
|---------|----------|-------|
| `check_baseline()` passes | `optimize_check_contract.py` | `baseline/` + workspace root |
| `check_round()` passes | `optimize_check_contract.py` | Current `opt-round-N/` |
| `--reset-optimize` | `resume.py::reset_optimize_workspace()` | Workspace root `*_result.pt` |
| Optimize session ends | `execution.py` finally blocks | Root + all `opt-round-*/` |
| `verify` completes | `verification/core.py::run_verify()` | `opt-verify/verify-*/` |

### Non-Modification Guarantee

All pt cleanup operates exclusively on `*_result.pt` and `test_result.pt` (case-insensitive). No other file types are touched. Cleanup failures in any trigger point are silently swallowed and do not affect the parent operation's result.

### Why Check Contracts Do Cleanup

The check contract modules (`check_baseline`, `check_round`) are the authority on what constitutes a valid baseline/round. They already verify that correctness status is recorded in the state JSON. Since the `.pt` archive contains no information beyond what the state JSON already captures, it is appropriate for the check contract to clean it after validation passes — keeping the cleanup alongside the validation that makes it safe.

## Impact on Verify

### Does verify need `.pt` files after completion?

No. The test correctness outcome is recorded in `verify-state.json` under `verify-result.test.status`. The `.pt` file path is also recorded in `result_artifact` but verify never reads `.pt` files from prior runs — it always re-executes tests fresh. `verify --reuse` relies on `verify-state.json` existence, not on `.pt` availability.

### If verify re-runs, does it re-generate `.pt`?

Yes. Each `verify` run executes the test harness from scratch, producing a new `*_result.pt` in the `opt-verify/verify-YYYYMMDD-HHMMSS/` directory. The cleanup at the end of `run_verify()` removes this file after `verify-state.json` is written.

### If user wants to keep the `.pt` for debugging?

The state JSON files (`baseline/state.json`, `round-state.json`, `verify-state.json`) preserve the essential outcome data (correctness_status, return codes, logs). The `.pt` files only contain raw tensor dumps useful for deep debugging. If those tensors are truly needed, the user can re-run the test harness to regenerate the `.pt`.

## Files Changed

| File | Change |
|------|--------|
| `src/triton_agent/optimize/pt_cleanup.py` | **New** — shared cleanup functions |
| `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py` | Import `cleanup_dir_pt_files`; call in `check_baseline()` and `check_round()` |
| `src/triton_agent/optimize/execution.py` | Import `cleanup_workspace_pt_files`; call in both finally blocks |
| `src/triton_agent/optimize/resume.py` | Add `*_result.pt` to `reset_optimize_workspace()` deletion list |
| `src/triton_agent/verification/core.py` | Import `cleanup_dir_pt_files`; call after `_write_verify_state()` |
