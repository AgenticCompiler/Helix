# TileLang Capture IR Guidance Design

## Summary

- Clarify the TileLang `capture_ir.py` preconditions in `skills/tilelang/tilelang-npu-analyze-ir/SKILL.md`, especially the need for module-level `@tilelang.jit` compilation and a non-underscore exported compiled-kernel variable.
- Add an explicit cache-cleanup step to the documented workflow so stale `__pycache__` and TileLang memoize state do not block recompilation after a failed JIT attempt.
- Improve `skills/tilelang/tilelang-npu-analyze-ir/scripts/capture_ir.py` error messages so the common failure modes point directly to the required operator edits and cache cleanup actions.
- Correct the TileLang optimize artifact reference so it describes AscendC source capture rather than the Triton/Bisheng IR archive layout.

## Problem

- The TileLang capture helper discovers compiled kernels by importing the operator module and scanning module-level names for objects that expose `get_kernel_source()`.
- If the operator file only defines `@tilelang.jit` factories but does not call them during module import, the helper finds no compiled kernels and returns a vague failure.
- If the compiled kernel is stored in a module-level name that starts with `_`, the helper skips it by design, which makes the failure look identical to the missing-call case.
- After a failed JIT compilation, stale Python and TileLang memoization caches can preserve the failed state, so a later capture attempt with the same parameters may fail again without re-running compilation.
- The current TileLang optimize artifact reference reuses Triton-oriented IR wording, which is misleading because the TileLang skill captures generated AscendC source instead of a Triton/Bisheng IR archive tree.

## Goals

- Document the exact operator preconditions needed for successful TileLang source capture.
- Add a standard cache-cleanup step before capture.
- Make the helper's common failures actionable without changing the CLI shape.
- Align TileLang optimize artifact documentation with the actual AscendC capture workflow.

## Non-Goals

- Do not add automatic cache deletion to `capture_ir.py`.
- Do not change the TileLang capture helper's CLI arguments or output format beyond clearer error text.
- Do not redesign the TileLang optimize workflow outside the specific TileLang IR-evidence guidance that points to this helper.

## Design

### 1. Skill documentation

Update `skills/tilelang/tilelang-npu-analyze-ir/SKILL.md` so the default workflow starts with explicit prerequisites:

- The operator file must trigger `@tilelang.jit` compilation during module import via a module-level call such as `compiled_kernel = kernel_func(...)`.
- The exported compiled-kernel variable must not start with `_` because the helper intentionally ignores underscore-prefixed module members.
- Call out explicitly that `_ = kernel_func(...)` may trigger compilation but is still the wrong documented pattern for this workflow because the helper will skip `_` during kernel discovery.

Add a cache-cleanup step before capture with an explicit workspace command:

```bash
find <workspace> -name "__pycache__" -type d -exec rm -rf {} +
```

The same section should also tell the user to remove stale TileLang memoization state, including `.pkl_memoize_py3`, when recompilation appears stuck on an old failed result.

Add a short troubleshooting table that maps these symptoms to causes and fixes:

- `No compiled kernels found`
- `No compiled kernels found` when the compiled object exists only under an underscore-prefixed name
- `Compilation Failed` or equivalent JIT/`get_kernel_source()` errors after a previous failed attempt

### 2. Script error guidance

Update `skills/tilelang/tilelang-npu-analyze-ir/scripts/capture_ir.py` so:

- the "no kernels found" path explicitly mentions both required fixes:
  - add a module-level factory call that runs during import
  - expose the compiled kernel through a name that does not start with `_`
- operator import failures and `get_kernel_source()` failures include cache-oriented repair guidance:
  - clear workspace `__pycache__`
  - clear stale `.pkl_memoize_py3` memoization artifacts
  - retry after forcing the module import path to recompile

Keep the helper's current behavior of failing explicitly instead of mutating the workspace automatically.

### 3. Optimize artifact reference correction

Update `skills/tilelang/tilelang-npu-optimize/references/artifacts.md` so the TileLang round-local evidence description refers to the TileLang AscendC source capture helper and examples such as:

```text
python3 ./scripts/capture_ir.py --operator-file opt-round-N/opt_<operator>.py > opt-round-N/ir/ascendc_source.cpp
```

The surrounding prose should describe round-local AscendC source capture and comparison, not Triton dump directories or Bisheng stage archives.

## Verification

- Add focused tests for the TileLang capture helper if the repository has a natural test home for skill-local script behavior; otherwise keep the implementation small and rely on direct script checks for this change.
- Run `bash scripts/run-skill-script-pyright.sh skills/tilelang/tilelang-npu-analyze-ir/scripts/capture_ir.py`.
- Run the smallest relevant regression tests that cover TileLang skill catalog or prompt-contract references if any touched documentation surfaces affect them.
