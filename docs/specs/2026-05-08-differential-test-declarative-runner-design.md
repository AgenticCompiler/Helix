# Declarative Differential Test Runner Design

## Summary

Differential `run-test` harnesses should become declarative import-only modules, similar to standalone benchmark files, while preserving the result payload format and `run-test` command interface.

## Goals

- Keep `run-test --test-file ... --operator-file ... --test-mode differential` as the public execution path.
- Keep the result payload format unchanged.
- Move differential case construction and execution into the test runner.
- Keep the generated differential test file import-only, with no direct CLI main flow.
- Preserve compatibility with existing script-style differential tests when possible.

## Decision

- Generated differential tests should export:
  - `build_operator_api(operator_module)`
  - `build_differential_test_cases(operator_api)`
- The runner should:
  - load the test module and operator module by file path
  - call `build_operator_api(operator_module)`
  - call `build_differential_test_cases(operator_api)`
  - execute each declared case in order
  - write the differential payload directly to `<operator>_result.pt`
- If a differential test module does not expose the declarative hooks, the runner may fall back to the legacy script-style execution path.
- The runner should still archive the differential result to `<operator>_result.pt` after a successful differential run, so `compare-result` continues to work unchanged.

## Verification

- Add tests for declarative differential loading and result writing.
- Add tests that legacy script-style differential tests still run through the fallback path.
- Update the differential test generation contract to require the import-only hook format.
