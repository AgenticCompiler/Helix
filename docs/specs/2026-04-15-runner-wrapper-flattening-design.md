# Runner Wrapper Flattening Design

## Summary

- Remove the redundant `src/triton_agent/test_runner.py` and `src/triton_agent/bench_runner.py` package wrappers.
- Keep execution-specific normalization in `src/triton_agent/execution.py`.
- Move optimize-only perf parsing to optimize-local code instead of routing through top-level wrapper modules.

## Goals

- Delete internal wrapper modules that only forward to triton-npu-run-eval skill scripts.
- Keep metadata parsing routed through the executable app layer instead of preserving fake package APIs.
- Keep optimize-only helper logic inside `src/triton_agent/optimize/`.

## Non-Goals

- Do not merge `execution.py` into `commands/execution.py`.
- Do not change local or remote run semantics.
- Do not rewrite skill scripts or move execution logic out of `skills/triton-npu-run-eval/scripts/`.

## Design

- Delete `src/triton_agent/test_runner.py`.
- Delete `src/triton_agent/bench_runner.py`.
- Update optimize resume logic to import metadata parsers from `src/triton_agent/execution.py`.
- Update optimize status logic to load perf parsing directly from the triton-npu-run-eval bench skill within optimize-local code.

## Verification

- Run `uv run python -m unittest tests.test_run_skill_loader tests.test_optimize_status tests.test_cli -v`
- Run `uv run --group dev ruff check`
- Run `uv run pyright`
