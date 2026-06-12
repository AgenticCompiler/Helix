# NPU Operator Accuracy Comparison Design

## Summary

Replace the current generic result-comparison flow with a single NPU-operator accuracy contract that applies to both `standalone` and `differential` test modes.

## Problem

The current correctness flow mixes two incompatible models:

- `standalone` tests compare outputs inline with ad hoc assertions such as `torch.testing.assert_close`
- `differential` tests archive only ordered outputs and rely on a repository-side `compare-result --compare-level ...` tolerance layer

That split is no longer sufficient for the new NPU operator comparison process because the new authority depends on:

- `--non-compute`
- inferred input tensor dtype family
- output dtype
- pre-check ordering for shape, NaN, and Inf handling
- floating-point thresholds that vary by output dtype
- richer failure diagnostics than a single max-diff summary

The old `strict|balanced|relaxed` surface also conflicts with the new rule set, which is now the only authority.

## Goals

- Make the new NPU operator comparison rule set the only correctness authority.
- Use one shared comparison implementation for both `standalone` and `differential` flows.
- Remove the legacy `--compare-level` interface and all logic behind it.
- Prevent generated standalone tests from being executed via `python test_xxx.py`.
- Keep the CLI thin: runner/orchestration code should own execution wiring, while comparison semantics stay in shared skill-side runtime code.
- Produce detailed failure diagnostics that tell an agent which case failed, which comparison path was selected, and exactly which check failed.

## Non-Goals

- Do not redesign benchmark, profile, or performance-comparison flows.
- Do not introduce task-file-structured input specs as a prerequisite for comparison.
- Do not support case-level `compute` overrides in this change; file-level metadata is sufficient.

## User-Visible Semantics

### One accuracy authority

All NPU operator correctness validation must use a single shared comparison implementation that follows the new rule set exactly.

No other threshold source remains valid:

- generated standalone tests must not use `torch.testing.assert_close`
- `compare-result` must not accept `strict|balanced|relaxed`
- `run-test` and `convert` differential validation must not expose `--compare-level`

### `# compute:` metadata

Generated test files must support a new header field:

```python
# compute: true
```

Rules:

- accepted values are `true` or `false`, case-insensitive after trimming
- missing metadata defaults to `true`
- this metadata is file-level and applies to all cases in that test file

The shared comparison implementation uses this flag as the source of `--non-compute` semantics:

- `compute: false` means non-compute path
- `compute: true` means compute path

### Standalone test contract

Generated standalone tests remain importable Python modules, but they are no longer self-executing scripts.

The standalone spec must require:

- a file-level metadata header including `# compute: ...`
- a `def main():` entrypoint
- no `if __name__ == "__main__": ...` block
- no direct `python test_xxx.py` execution contract
- use of the shared comparison helper instead of inline `assert_close`

The standalone test module continues to own:

- parsing metadata/constants embedded in the file
- loading the runtime operator module from the target operator path
- constructing deterministic NPU test inputs
- computing the PyTorch golden output
- calling the shared comparison helper on each case

The runner owns:

- importing the test module
- preparing import paths/environment so the shared comparison helper can be imported
- preparing `sys.argv` so the existing `--operator-file` parsing contract still works when `main()` is called programmatically
- calling `main()`

### Differential test contract

Generated differential tests remain import-only declarative modules.

They must continue to export:

- `build_operator_api(operator_module)`
- `build_differential_test_cases(operator_api)`

The returned cases must be upgraded so the runner can archive comparison context, not only final outputs. Each case result contract should support the runner capturing:

- case id
- case inputs
- operator output

The compare flow must then use the archived inputs plus golden output to apply the same rule set as standalone mode.

### Detailed diagnostics

Comparison failures must be explicit enough for an agent to repair the operator without re-deriving the decision path from source code.

At minimum, a failing comparison should report:

- case id
- whether the case was treated as compute or non-compute
- inferred input classification: `float`, `int`, or `no_tensor`
- effective output dtype
- selected decision path:
  - non-compute
  - bool output
  - integer compute
  - quantized fp-to-int
  - floating-point compute
- the first failed pre-check, if any
- for floating-point failures, which AND clause failed:
  - max error cap
  - matched ratio
  - MERE
- relevant observed values such as shape, dtype, mismatch counts, worst offending index/value, and active thresholds

Passing output can stay concise, but the structured result object should still retain enough detail for future logging or reporting.

## Design

### Shared comparison runtime

Add a single shared comparison module under `skills/triton-npu-run-eval/scripts/` that implements the new NPU operator accuracy contract.

This module should expose:

- a case-level comparison entrypoint for direct harness use
- an artifact-level comparison entrypoint for archived differential payloads
- structured compare result objects with:
  - `passed`
  - `case_id`
  - `comparison_path`
  - summary message
  - detailed metrics/diagnostics

The same implementation must power:

- standalone inline validation in generated tests
- differential archive comparison in `compare-result`
- automatic compare steps used by `run-test` and `convert`

### Input-type inference

The comparison runtime must infer input type from actual runtime objects only, not from a structured test spec.

Rules:

1. If any `torch.Tensor` exists in the input tree, choose the highest-priority dtype among all tensors.
2. Otherwise, if a list/tuple of tensors exists, use the first tensor-list element dtype.
3. Otherwise treat the case as `no_tensor`.

This inference must then classify input type as:

- `float`
- `int`
- `no_tensor`

The bool-input and bool-output special handling must follow the provided authority exactly.

### Decision matrix implementation

The runtime must implement the five-path decision matrix exactly:

- non-compute
- bool output
- integer compute
- quantized float-to-int
- floating-point compute

The implementation should normalize decision selection into an explicit enum/string so diagnostics can report the chosen path directly.

### Ordered pre-checks

Before numeric comparison, the runtime must perform the pre-checks in the documented order:

1. shape equality
2. NaN mask equality
3. Inf mask/sign equality
4. bool equality shortcut

Only finite finite pairs enter numeric comparison.

When tensor dtypes differ for finite comparison, the implementation side is cast to the golden dtype before numeric checks.

### Floating-point comparison details

Floating-point comparison must implement all three required clauses and require all three to pass:

- max error cap
- matched ratio
- MERE

Threshold resolution must be driven by output dtype, including explicit handling for:

- `float16`
- `bfloat16`
- `float32`
- `hifloat32`
- `float8_e4m3`
- `float8_e5m2`
- fallback

The implementation should centralize these threshold tables in one place and make the selected threshold row available in diagnostics output.

### Standalone runner refactor

`run-test` local and remote standalone execution must stop shelling out with:

```bash
python test_xxx.py --operator-file ...
```

Instead, the runner should:

1. import the test module by path
2. prepare environment/import paths needed by the shared comparison helper
3. temporarily set `sys.argv` to the equivalent of `test_xxx.py --operator-file <path>`
4. call `main()`

This import-and-call contract is the mechanism that prevents accidental direct script execution by downstream agents.

Legacy standalone script-style execution should be removed instead of preserved as compatibility behavior.

### Differential payload upgrade

The current differential archive format only stores:

```python
{"results": [...]}
```

That is insufficient because the new rule set depends on actual inputs.

The archived payload must be upgraded to include per-case records with enough data to compare under the shared rule set. A minimal shape is:

```python
{
    "cases": [
        {
            "id": "...",
            "inputs": ...,
            "result": ...,
        }
    ]
}
```

Additional metadata may be stored if it keeps the runner simpler, but the payload should not grow unrelated fields.

The compare implementation should treat the oracle payload as the golden source for:

- case ordering
- case ids
- case inputs
- expected results

The candidate payload supplies candidate results for the same case ids/order.

### Metadata handling

`parse_test_metadata()` should be extended so `compute` is available anywhere run-test orchestration or compare code needs it.

Comparison callers should resolve compute semantics as:

- explicit parsed metadata when present
- default `true` when absent

The comparison runtime should receive a normalized boolean rather than re-parsing raw strings in multiple places.

### CLI and skill-surface cleanup

Remove `--compare-level` from:

- repository CLI parsing
- run-eval helper script parsing
- skill docs
- README examples
- tests that lock old parser/help behavior

After this change:

- `compare-result` always uses the shared NPU comparison rule set
- `run-test` differential comparison always uses the shared NPU comparison rule set
- `convert` differential verification always uses the shared NPU comparison rule set

No compatibility alias should remain for `strict|balanced|relaxed`.

## Failure Reporting Contract

Comparison failures should expose both machine-usable structure and human-readable summaries.

Recommended result shape:

- top-level pass/fail
- failed case count
- per-case entries
- per-case message
- per-case diagnostics payload

Recommended diagnostics fields include:

- `case_id`
- `compute`
- `input_type`
- `input_dtype`
- `output_dtype`
- `comparison_path`
- `failure_stage`
- `shape_expected`
- `shape_actual`
- `nan_mismatch_count`
- `inf_mismatch_count`
- `finite_count`
- `matched_ratio`
- `mere`
- `mere_threshold`
- `max_abs_diff`
- `max_abs_diff_index`
- `max_error_cap_at_index`
- selected threshold row

The CLI should print a concise summary plus enough case detail to guide repair. The full structured details should remain available to callers or future JSON/reporting extensions.

## Files Likely To Change

| File | Change |
|------|--------|
| `docs/specs/2026-06-12-npu-operator-accuracy-comparison-design.md` | Record the new comparison authority and runner contracts |
| `skills/triton-npu-run-eval/scripts/compare_result.py` | Replace old tolerance-level logic with shared NPU comparison over archived payloads |
| `skills/triton-npu-run-eval/scripts/test_runner.py` | Import-and-call standalone tests, upgrade differential archiving, and route both modes through shared compare helpers |
| `skills/triton-npu-run-eval/scripts/run-command.py` | Remove `--compare-level` from helper CLI surfaces |
| `src/triton_agent/commands/comparison.py` | Remove compare-level plumbing |
| `src/triton_agent/commands/execution.py` | Remove compare-level plumbing and keep automatic differential compare on the shared rule set |
| `src/triton_agent/commands/convert.py` | Remove compare-level plumbing and keep convert verification on the shared rule set |
| `skills/triton-npu-gen-test/references/test-standalone-spec.md` | Require import-only standalone structure, `# compute:`, shared compare helper, and non-self-executing `main()` |
| `skills/triton-npu-gen-test/references/test-differential-spec.md` | Require `# compute:` and declarative case contract that supports archiving inputs plus results |
| `skills/triton-npu-run-eval/references/run-test.md` | Remove compare-level wording and document shared compare behavior |
| `skills/triton-npu-run-eval/references/compare-result.md` | Remove compare-level wording and describe new comparison semantics |
| `README.md` | Remove compare-level flags and update run-test/compare-result docs |
| `tests/test_test_runner.py` | Cover standalone import-and-call flow, differential payload upgrade, and diagnostics behavior |
| `tests/test_comparison_commands.py` | Remove compare-level expectations |
| `tests/test_execution_commands.py` | Remove compare-level expectations and preserve differential auto-compare flow |
| `tests/test_convert_commands.py` | Remove compare-level expectations and preserve convert verification flow |
| `tests/test_skill_command_script.py` | Remove compare-level parser/help expectations |
| `tests/test_generation_contracts.py` | Lock the new standalone/differential generation contracts |

## Verification

- Focused unit tests for `compare_result.py` covering all five decision paths and the ordered pre-checks.
- Focused runner tests for:
  - standalone import-and-call execution
  - rejection of self-executing standalone contract in generated docs/tests
  - differential payload archiving with inputs and results
  - detailed failure diagnostics
- Focused command tests proving `--compare-level` is removed from parser/help/dispatch.
- Contract tests for updated test generation specs and run-eval docs.
- `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/compare_result.py`
- `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/test_runner.py`
- Repository verification:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
