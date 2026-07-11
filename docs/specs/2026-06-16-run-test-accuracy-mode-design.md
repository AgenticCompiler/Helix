# Run-Test Accuracy Mode Design

## Summary

- Add `--accuracy-mode npu-contract|dtype-close` to `run-test` and `compare-result`.
- Keep `npu-contract` as the default so existing behavior stays unchanged unless the user opts in.
- Treat `npu_compare` as the only public comparison API, and let it dispatch internally to either the existing NPU contract logic or a new dtype-aware `assert_close` style path.

## Goals

- Let users choose between the current NPU accuracy contract and a dtype-aware close-comparison mode from the run/evaluate entrypoints.
- Apply the same selection to both standalone inline validation and differential archived-result comparison.
- Reuse one public comparison surface instead of introducing parallel comparison modules.
- Keep existing generated standalone and differential test-file contracts valid.

## Non-Goals

- Do not restore `--compare-level strict|balanced|relaxed`.
- Do not change benchmark, profiling, or performance-comparison flows.
- Do not require users to regenerate existing test harnesses just to use `dtype-close`.
- Do not add a new MCP `compare-result` tool in this change; the current MCP surface only exposes `run-test-baseline` and `run-test-optimize`.

## User-Visible Behavior

### CLI

`run-test` accepts:

```bash
--accuracy-mode npu-contract|dtype-close
```

Rules:

- Default is `npu-contract`.
- The flag applies to both `standalone` and `differential` test modes.
- In `differential` mode, it affects automatic archived-result comparison triggered by `--ref-result` or `--ref-operator-file`.

`compare-result` accepts the same flag with the same default:

```bash
--accuracy-mode npu-contract|dtype-close
```

This keeps manual comparison behavior aligned with `run-test`.

Examples:

```bash
uv run helix run-test \
  --test-file test_abs.py \
  --operator-file abs.py
```

```bash
uv run helix run-test \
  --test-file test_abs.py \
  --operator-file abs.py \
  --accuracy-mode dtype-close
```

```bash
uv run helix compare-result \
  --ref-result abs_result.pt \
  --new-result opt_abs_result.pt \
  --accuracy-mode dtype-close
```

### MCP

The managed run-eval MCP server currently exposes:

- `run-test-baseline`
- `run-test-optimize`

Do not expose `accuracy_mode`, `atol`, or `rtol` as MCP tool parameters. Agents should not actively choose the precision policy. Instead, configure the managed run-eval MCP server process with `HELIX_RUN_TEST_ACCURACY_MODE`, `HELIX_RUN_TEST_ATOL`, and `HELIX_RUN_TEST_RTOL` before the agent calls `run-test-baseline` or `run-test-optimize`; the spawned staged run-eval CLI inherits those environment variables.

This change does not add a new MCP `compare-result` tool.

## Comparison Semantics

### Public API Shape

`skills/common/ascend-npu-run-eval/scripts/npu_compare.py` remains the only public comparison module.

Its public entrypoints should become:

- `compare_case_result(..., accuracy_mode: str | None = None)`
- `compare_result_payloads(..., accuracy_mode: str | None = None)`

Behavior:

- If `accuracy_mode` is provided explicitly, use it.
- Otherwise resolve the mode from runner-owned shared runtime configuration.
- If no explicit or runtime-provided value exists, default to `npu-contract`.

This keeps existing generated harnesses valid because they already call `compare_case_result(...)` without an accuracy-mode argument.

### Runtime configuration mechanism

Agent-facing run-eval execution uses environment variables as the runner-owned runtime configuration mechanism. This is required because the staged run-eval CLI launches local worker subprocesses and remote SSH commands; environment variables cross those process boundaries without changing generated harness contracts.

The environment variables are:

- `HELIX_RUN_TEST_ACCURACY_MODE`: optional `npu-contract` or `dtype-close`; default is `npu-contract`
- `HELIX_RUN_TEST_ATOL`: optional dtype-close absolute tolerance override
- `HELIX_RUN_TEST_RTOL`: optional dtype-close relative tolerance override

`npu_compare.py` should resolve accuracy mode in this order:

1. explicit `accuracy_mode` argument
2. `HELIX_RUN_TEST_ACCURACY_MODE`
3. `"npu-contract"`

`HELIX_RUN_TEST_ATOL` and `HELIX_RUN_TEST_RTOL` should only affect the `dtype-close` floating-point and complex comparison path.

The top-level `helix run-test` command still exposes an explicit `--accuracy-mode` option and forwards that value into the skill runner environment. This keeps the top-level CLI deterministic even if the user's shell already has run-eval environment variables set.

Harnesses may import `npu_compare` before the environment is read. That is acceptable because accuracy mode and tolerance are resolved when `compare_case_result(...)` executes, not at import time.

### `npu-contract`

`npu-contract` keeps the current NPU accuracy contract unchanged:

- compute vs non-compute routing
- dtype-family-based decision paths
- current contract thresholds and diagnostics
- current artifact-comparison structure and failure semantics

Existing `_MATCH_THRESHOLDS` and `_MAX_ERROR_THRESHOLDS` tables remain unchanged for the `npu-contract` path.

This is the compatibility default and remains the authoritative mode for current users unless they opt into `dtype-close`.

### `dtype-close`

`dtype-close` is a new comparison mode designed to feel close to the historical `balanced` behavior, but without reviving the `compare-level` surface.

It should behave as follows:

- payload shape, case count, case order, case id, and file-level compute flag remain strict
- mapping keys, sequence length, tensor shape, and tensor dtype remain strict
- non-compute outputs remain exact-equality checks
- bool outputs remain exact-equality checks
- integer outputs remain exact-equality checks
- floating-point and complex outputs use `torch.testing.assert_close(..., equal_nan=True)` with dtype-selected tolerances

The purpose of `dtype-close` is not to replicate the full NPU contract decision matrix. It is a simpler dtype-aware close-comparison mode that preserves structural strictness and uses explicit tolerance tables only for floating-point-like numeric leaves.

### Leaf dispatch architecture

`dtype-close` should reuse the existing leaf-routing structure instead of introducing a second top-level comparison tree.

Specifically:

- `_compare_leaf(...)` should continue to own recursive structure handling
- `_select_comparison_path(...)` should continue to determine the path name
- existing prechecks for shape, NaN masks, Inf masks, and Inf sign/value agreement should stay shared
- `_compare_non_compute(...)`, `_compare_bool_output(...)`, and `_compare_integer_output(...)` should remain the exact-equality implementations for both accuracy modes
- only the floating-point leaf comparator should branch on `accuracy_mode`

This avoids duplicating the structural validation logic and keeps `dtype-close` scoped to the part of the behavior that is actually changing.

Artifact comparison should continue delegating each matched case through `compare_case_result(...)` so standalone and differential validation share the same per-case semantics.

### `assert_close` exception translation

`torch.testing.assert_close(...)` raises `AssertionError` on mismatch. `npu_compare` must not let that raw exception escape through its public comparison API.

Instead, the dtype-close floating-point path should:

- call `torch.testing.assert_close(...)` inside a `try` block
- return a passing `CaseCompareResult` when no exception is raised
- catch `AssertionError`
- convert it into a failing `CaseCompareResult`

The failing result should preserve structured diagnostics, including:

- `accuracy_mode`
- `comparison_path`
- `output_dtype`
- tensor shape
- selected `rtol`
- selected `atol`
- the rendered `AssertionError` summary under a stable diagnostics key such as `assert_close_message`
- optional `max_abs_diff` when cheaply available

`assert_close_message` should be treated as a stable diagnostics-contract key once introduced so downstream tooling can rely on it.

## Dtype-Close Tolerance Table

`dtype-close` should use a new dedicated tolerance table instead of reusing the current `npu-contract` thresholds.

The default table should be:

| Dtype family | `rtol` | `atol` |
| --- | ---: | ---: |
| `float64` | `1e-5` | `1e-8` |
| `complex128` | `1e-5` | `1e-8` |
| `float32` | `1e-4` | `1e-5` |
| `hifloat32` | `1e-4` | `1e-5` |
| `complex64` | `1e-4` | `1e-5` |
| `float16` | `5e-4` | `5e-5` |
| `bfloat16` | `1e-3` | `1e-4` |
| `float8_e4m3*` | `1e-2` | `1e-3` |
| `float8_e5m2*` | `1e-2` | `1e-3` |
| fallback floating dtype | `1e-4` | `1e-5` |

Notes:

- The `float32` row is the anchor and intentionally matches the old `balanced` feel.
- `hifloat32` should be treated explicitly instead of falling through to the generic fallback row.
- `float16` and `bfloat16` are only moderately looser than `float32`; they must not inherit the much looser `npu-contract` max-error thresholds.
- Complex dtypes reuse the tolerance of the corresponding real-precision family.
- Integer, bool, and non-compute paths do not consult this table.

## Standalone Flow

Generated standalone tests should continue to use:

```python
from npu_compare import compare_case_result
```

The generated file contract does not change in this design:

- no new required metadata field
- no new required helper import
- no required `accuracy_mode=` argument in generated harness code

Instead, runner-owned execution should provide the selected accuracy mode as environment-backed shared runtime configuration before the test module calls `compare_case_result(...)`.

More precisely:

- import the harness module normally
- resolve `operator_api`
- set `HELIX_RUN_TEST_ACCURACY_MODE` for the local worker or remote command environment
- call `main(operator_api)`
- avoid adding any generated-harness argument or required metadata field

The environment-backed configuration is scoped to the launched worker process or remote command. Harness code should not change it mid-run.

This preserves compatibility with already-generated test files.

## Differential Flow

Differential archived-result comparison should also route through `npu_compare`.

Behavior by mode:

- `npu-contract`: keep the current `compare_result_payloads(...)` semantics unchanged
- `dtype-close`: reuse the same artifact structure validation, then compare each matching case result with dtype-close semantics instead of the NPU contract path

This means `compare-result` and `run-test --ref-result/--ref-operator-file` remain structurally strict even in `dtype-close`.

For differential comparison, the mode should be passed explicitly into `compare_result_payloads(...)`; it does not need the standalone context-local override mechanism.

No extra artifact-level comparison tree should be introduced here. The existing artifact flow should continue delegating matched cases through `compare_case_result(...)`, with `accuracy_mode` threaded into that existing per-case path.

## Runtime Wiring

### CLI and command layer

Update these entrypoints to accept and forward `accuracy_mode`:

- `src/helix/cli.py`
- `src/helix/commands/execution.py`
- `src/helix/commands/comparison.py`
- `skills/common/ascend-npu-run-eval/scripts/cli.py`

Rules:

- `run-test` parser accepts `--accuracy-mode`
- `compare-result` parser accepts `--accuracy-mode`
- `run-test` forwards the selected mode into both standalone and differential compare paths
- `compare-result` forwards the selected mode into the shared compare runtime
- `CompareResultModule` protocol methods and their package-side wrappers must gain an `accuracy_mode` parameter
- `handle_run_test(...)` must thread `accuracy_mode` into its automatic differential compare call instead of calling `compare_result_files(ref_result, archived_result)` without the selected mode

### Runner and remote execution

Update these skill-side runtime files:

- `skills/common/ascend-npu-run-eval/scripts/test_runner.py`
- `skills/common/ascend-npu-run-eval/scripts/compare_result.py`
- `skills/common/ascend-npu-run-eval/scripts/npu_compare.py`

Requirements:

- local standalone runs must see the selected mode
- remote standalone runs must see the selected mode
- local differential compare must see the selected mode
- remote differential compare must see the selected mode

Implementation notes:

- standalone runners should use the context-local override mechanism described above
- `compare_result.py` local comparison should pass `accuracy_mode` explicitly into `compare_result_payloads(...)`
- `compare_result.py` remote comparison should append `--accuracy-mode <mode>` to the remote `python3 compare_result.py ...` invocation
- automatic differential comparison launched from `run-test` should forward the selected mode through the same explicit compare-result helper path

### MCP forwarding

Update:

- `src/helix/eval/mcp_server.py`

Requirements:

- `run-test-baseline` does not accept `accuracy_mode`, `atol`, or `rtol`
- `run-test-optimize` does not accept `accuracy_mode`, `atol`, or `rtol`
- the MCP server should not synthesize accuracy environment variables from tool arguments
- staged run-eval subprocesses inherit the managed MCP server process environment normally
- tool metadata must not document precision controls as agent-visible parameters

## Diagnostics

All failure messages should include the effective `accuracy_mode`.

For `dtype-close`, tensor mismatch diagnostics should include at least:

- comparison path
- tensor dtype
- tensor shape
- selected `rtol`
- selected `atol`
- the underlying `assert_close` mismatch summary

When available at low implementation cost, include `max_abs_diff` as an extra diagnostic field.

For exact-equality paths reused by `dtype-close`:

- non-compute
- bool-output
- integer-compute
- quantized-fp-to-int

the result shape may stay the same as today, but diagnostics should still expose the effective `accuracy_mode`.

## Documentation Scope

Update:

- `README.md`
- `skills/common/ascend-npu-run-eval/references/run-test.md`

Remove:

- `skills/common/ascend-npu-run-eval/references/compare-result.md`, because the staged run-eval `cli.py` no longer exposes an agent-facing `compare-result` subcommand

Do not update:

- `skills/triton-npu-gen-test/references/test-standalone-spec.md`

Reason:

- the harness contract does not change
- only runtime comparison-mode selection changes

## Verification

Add or update focused tests for:

- parser support and defaults for `run-test --accuracy-mode`
- parser support and defaults for `compare-result --accuracy-mode`
- forwarding from command handlers to skill-side wrappers
- forwarding from MCP tool arguments to `run-test-*` subcommands
- `npu_compare` defaulting to `npu-contract`
- `npu_compare` explicit `dtype-close` case comparison
- `npu_compare` explicit `dtype-close` artifact comparison
- strict structural mismatch behavior remaining intact under `dtype-close`
- remote `run-test` and remote `compare-result` forwarding of the selected mode

Expected verification commands after implementation:

- `uv run python -m unittest tests.test_execution_commands -v`
- `uv run python -m unittest tests.test_comparison_commands -v`
- `uv run python -m unittest tests.test_npu_compare -v`
- `uv run python -m unittest tests.test_skill_command_script -v`
- `uv run python -m unittest tests.test_run_eval_mcp_server_tool_metadata -v`
- `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/npu_compare.py`
- `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/compare_result.py`
- `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/test_runner.py`
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
