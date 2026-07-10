# Run-Test Case-Id Compare-Result Loader Design

## Summary

Fix local `run-test --case-id ... --ref-operator-file ...` comparisons so they can load the bundled `npu_compare.py` helper without relying on a transient `sys.path` side effect from the skill loader.

## Problem

- The `--case-id` execution path compares in-memory payload objects directly after the reference and candidate case runs finish.
- That path reaches `skills/common/ascend-npu-run-eval/scripts/compare_result.py`, which currently does `importlib.import_module("npu_compare")`.
- The shared skill loader only adds the skill `scripts/` directory to `sys.path` while loading `compare_result.py` itself, then removes it again.
- As a result, the later runtime import of `npu_compare` can fail with `ModuleNotFoundError` even though the helper lives beside `compare_result.py`.

## Approach

- Keep the public comparison behavior unchanged.
- Add a small local module-loading helper inside `compare_result.py` that loads `npu_compare.py` from the same directory as the script.
- Cache the loaded module so repeated comparisons in one process do not reload the helper unnecessarily.
- Do not expand this change into remote helper staging; this fix is scoped to the local loader regression.

## Verification

- Add a regression test that calls the top-level comparison wrapper with in-memory payload objects and proves the bundled helper loads successfully.
- Run the focused comparison test file.
- Run strict pyright for the modified skill script.
- Run repo-level `ruff` and `pyright`.
