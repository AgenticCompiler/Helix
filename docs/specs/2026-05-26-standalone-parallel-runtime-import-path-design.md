# Standalone Parallel Runtime Import Path Design

## Summary

Parallel `standalone` bench workers launch `python -c` subprocesses that import `standalone_bench_runtime` by module name. When the staged case workspace preserves the benchmark's nested skill layout, that runtime file lives under `.opencode/skills/.../scripts/` instead of the workspace root, so the import fails with `ModuleNotFoundError`.

## Goals

- Keep parallel `standalone` workers working when support files are staged under nested skill paths.
- Preserve the existing case workspace copy layout and relative benchmark/operator paths.
- Add a regression test that exercises the nested support-path layout.

## Non-Goals

- Do not flatten staged support files into the workspace root.
- Do not change sequential `standalone` execution or `msprof` execution semantics.
- Do not add new CLI flags or user-facing behavior.

## Decision

- Keep copying support files with their existing relative paths.
- Update the generated parallel worker `python -c` script to prepend the staged runtime script directory to `sys.path` before importing `standalone_bench_runtime`.
- Resolve that runtime directory from the staged workspace-relative path of `standalone_bench_runtime.py` so the same logic works for both flattened and nested layouts.

## Verification

- Add a focused unit test that stages `standalone_bench_runtime.py` only under a nested `.opencode/skills/.../scripts/` path and verifies parallel `standalone` execution succeeds.
- Run the focused bench runner unit tests that cover parallel `standalone` execution.
- Run the file-scoped strict `pyright` check for `skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`.
