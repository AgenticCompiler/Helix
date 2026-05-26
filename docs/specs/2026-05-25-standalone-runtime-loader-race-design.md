# Standalone Runtime Loader Race Design

## Goal

Prevent standalone benchmark parallel mode from failing during runtime-helper import with `AttributeError: 'NoneType' object has no attribute '__dict__'`.

## Problem

`bench_runner.py` dynamically loads `standalone_bench_runtime.py` through `_load_standalone_runtime_module()`. The current loader uses one fixed module name, inserts that module into `sys.modules`, executes it, and then removes it again.

In standalone parallel mode, multiple worker threads can trigger that helper load at nearly the same time while preparing per-case workspaces. If one thread removes the shared `sys.modules` entry while another thread is still executing `standalone_bench_runtime.py`, `dataclasses` can no longer resolve `sys.modules[cls.__module__]` while processing `@dataclass`, and import fails.

## User-Visible Behavior

- `run-bench --bench-mode standalone --npu-devices ...` should work in parallel mode without intermittent import crashes.
- Serial standalone mode should keep its current behavior.
- The runtime helper API and staged support files should remain unchanged.

## Design

Keep the fix local to `skills/triton-npu-run-eval/scripts/bench_runner.py`.

- Add a process-local cache for the loaded standalone runtime module.
- Guard first-time initialization with a lock so only one thread executes the dynamic import.
- Return the cached module for all later calls, including worker-thread calls that only need `runtime_support_paths()`.
- Preserve the existing import error handling and module-name contract.

This removes the concurrent same-name import window without changing standalone runtime behavior or the worker orchestration flow.

## Testing

- Add a regression test that calls `_load_standalone_runtime_module()` from multiple threads while a fake loader intentionally pauses during `exec_module()`.
- Assert that only one loader execution occurs and every caller receives the same module object.
- Keep existing standalone parallel tests unchanged to confirm the public workflow still behaves the same.

## Scope

- Do not change standalone benchmark case semantics.
- Do not change remote staging inputs.
- Do not refactor runtime helper loading outside the standalone benchmark path.
