# Remote Differential Follow-Ups Design

## Summary

- Fix the remote differential runner so generated remote scripts can execute without missing helper definitions.
- Keep local and remote differential case validation aligned so the same declarative test contract is enforced in both environments.
- Reduce duplicated baseline-result resolution logic in `run-command.py` without changing CLI behavior, error text, or exit codes.

## User-Visible Semantics

- `run-test --test-mode differential --remote ...` must execute generated differential tests successfully when the test file follows the documented contract.
- Remote differential execution must honor `# compute-kind: compute|non-compute` the same way local execution does.
- Remote differential execution must accept the same case container contract as local execution:
  - the case collection must be an iterable, not a string/bytes/mapping
  - each case must be a mapping
  - each case must contain `id`, `inputs`, and `fn`
  - `inputs` must be a list or tuple
- Baseline result resolution for differential comparison must behave exactly as before for:
  - direct `--baseline-result`
  - derived result reuse when `<baseline>_result.pt` already exists
  - running a baseline operator locally or remotely when an archive must be produced

## Implementation Approach

### Remote differential helper completeness

- Extend the generated remote differential script so it includes every helper it calls, including `compute-kind` parsing.
- Keep the helper surface minimal and self-contained so the remote script still runs without importing the local repository package graph.

### Shared differential case validation semantics

- Define a small shared normalization helper in `test_runner.py` for remote differential archive materialization.
- Generate the remote script with that helper body so local and remote validation stay in sync from one source definition.
- Preserve the current local `DifferentialTestCase` contract and archive payload format.

### Baseline resolution cleanup

- Split `_resolve_run_test_comparison_inputs()` into:
  - mode-specific argument validation
  - one shared helper that reuses or produces the baseline archive
- Keep all existing parser errors, output rendering, and exit behavior unchanged.

## Non-Goals

- Do not redesign standalone execution.
- Do not change the differential archive schema.
- Do not add compatibility fallbacks for deprecated test contracts.
- Do not change comparison semantics in `npu_compare.py`.

## Verification

- Add regression tests that execute the generated remote differential script instead of only inspecting the command string.
- Run the targeted unit tests for `test_runner`, `run-command`, and remote execution flows.
- Run strict skill-script pyright checks for any modified files under `skills/*/scripts/`.
