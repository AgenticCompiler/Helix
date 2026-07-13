# Optimize Check PT Cleanup Compatibility Design

## Summary

Keep `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/` self-contained so staged optimize-check scripts do not import `helix` at all.

## Problem

`skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check_contract.py` had started importing helper logic from `helix.optimize.*`.

That violates the confirmed project boundary for this skill family: the optimize-check skill should own its script-side helper logic, while `src/helix` may load and call the skill through the bridge layer. Once the skill script imports `helix`, direct script execution becomes coupled to whichever runtime package happens to be present, and staged workspaces can fail before contract checks even start.

## Decision

Move the needed helpers back into `optimize_check_contract.py` itself.

- Define `cleanup_dir_pt_files()` locally in the skill script.
- Define round artifact naming and resolution helpers locally in the skill script instead of importing `helix.optimize.naming`.
- Preserve the current cleanup contract: only remove `test_result.pt` or `*_result.pt`, sort deterministically, and swallow `OSError`.
- Preserve the current round artifact compatibility behavior for expected `opt_*.py`, `opt_*_perf.txt`, and legacy fallback names.

## Non-Goals

- Do not change optimize/verify runtime cleanup behavior in `src/helix`.
- Do not refactor the broader `run-eval` bridge pattern in this fix.
- Do not remove the existing `src/helix/optimize/pt_cleanup.py` module, which is still used by runtime code.

## Verification

- Add a boundary regression asserting optimize-check skill scripts do not import `helix`.
- Run the required strict skill-script `pyright` check for `optimize_check_contract.py`.
