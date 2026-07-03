# NPU Compare Module Split Design

## Summary

Refactor `skills/common/ascend-npu-run-eval/scripts/npu_compare.py` into a thin public API module plus mode-oriented implementation modules. Keep the exported API and observable comparison behavior unchanged.

## Goals

- Preserve `npu_compare.py` as the public entrypoint used by tests and runtime code.
- Split the comparison implementation by accuracy semantics first:
  - `npu-contract`
  - `dtype-close`
- Keep files small and readable.
- Avoid duplicated threshold tables, result builders, and tensor helpers.

## Non-Goals

- No behavior changes to comparison rules, diagnostics, or public result types.
- No rename of public functions.
- No new accuracy modes.

## Proposed Structure

- `npu_compare.py`
  - public API
  - payload/case traversal
  - leaf routing
- `npu_compare_common.py`
  - shared dataclasses
  - accuracy-mode context and env parsing
  - shared dtype tables and tensor helpers
  - shared result builders
- `npu_contract_compare.py`
  - `npu-contract` comparison implementation
  - route by comparison path inside contract mode
- `dtype_close_compare.py`
  - `dtype-close` comparison implementation

If `npu_contract_compare.py` remains too large after extraction, follow up by splitting it into compute and non-compute helpers. That split is optional and should be driven by readability, not symmetry.

## Behavior

`compare_case_result()` and `compare_result_payloads()` continue to resolve the active accuracy mode exactly as today. Leaf comparison keeps the current output structure checks, dtype coercion, comparison path selection, and diagnostics.

For `npu-contract`, the implementation keeps:

- non-compute raw-bit equality
- bool output equality
- integer comparison with the existing bounds
- floating-point threshold logic
- shared prechecks for shape, NaN, and Inf handling

For `dtype-close`, the implementation keeps:

- strict dtype equality before comparison
- existing `torch.testing.assert_close()` tolerances
- environment overrides for `atol` and `rtol`

## Verification

- Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_npu_compare.py`
- Run `uv run --group dev ruff check`
- Run `uv run pyright`
