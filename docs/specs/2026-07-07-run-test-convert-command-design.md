# Run-Test Convert Command Design

## Summary

- Add `run-test-convert` to the skill-local `ascend-npu-run-eval` helper CLI.
- Expose the same convert-specific entrypoint from the managed run-eval MCP server.
- Keep `run-test-baseline` and `run-test-optimize` as the baseline/generation and optimize-round surfaces, and stop making convert guidance borrow those names.
- Tighten run-test option validation so convert and optimize differential flows fail fast when agents omit or over-specify reference inputs.
- Keep the baseline MCP tool intentionally narrower than the helper CLI while exposing differential reference inputs for convert and optimize MCP tools.

## Problem

Convert validation currently reuses `run-test-baseline` for standalone mode and `run-test-optimize` for differential mode. The implementation works, but the command names communicate the wrong workflow intent. That mismatch leaks into skill guidance: the convert skills have to explain why a convert workflow should call an optimize-named command.

At the same time, the helper CLI still permits option combinations that are only valid for one workflow shape. In particular, `run-test-baseline` differential mode intentionally allows no reference input so the command can produce a reusable archived baseline result. That flexibility is correct for baseline generation, but it is the wrong default for convert and optimize validation, where a missing reference means the correctness gate is incomplete. We should encode those workflow rules directly in the CLI contract so agent mistakes fail immediately.

## Goals

- Give convert workflows a dedicated run-eval command whose name matches convert semantics.
- Reject invalid `standalone`/`differential` reference-flag combinations before execution starts.
- Reuse the existing local/remote test runner and archived-result comparison backend.
- Preserve `run-test-baseline` differential mode as the one workflow that may intentionally run without a reference input in order to produce baseline evidence.

## Non-Goals

- Do not add a top-level repository command such as `triton-agent run-test-convert`.
- Do not redesign `src/triton_agent/commands/convert.py` to shell out through the skill-local helper CLI.
- Do not remove the existing `--baseline-operator-file` alias; keep it as a compatibility synonym for `--ref-operator-file`.
- Do not change benchmark, profiling, `compare-result`, or non-run-test execution flows.

## User-Visible Behavior

### Skill-Local CLI

Add a new helper subcommand:

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert ...
```

Example convert invocations:

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert \
  --test-file test_kernel.py \
  --operator-file triton_kernel.py \
  --test-mode standalone
```

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert \
  --test-file differential_test_kernel.py \
  --operator-file triton_kernel.py \
  --test-mode differential \
  --ref-operator-file kernel.py
```

`run-test-convert` accepts the same execution arguments as the existing run-test helper family:

- `--test-file`
- `--operator-file`
- `--ref-result`
- `--ref-operator-file`
- `--remote`
- `--remote-workdir`
- `--keep-remote-workdir`
- `--verbose`
- `--test-mode`

### Mode-Specific Validation Rules

After resolving `test-mode` from the explicit flag or from test metadata, the helper must apply these rules:

| Command | `standalone` | `differential` |
| --- | --- | --- |
| `run-test-baseline` | reject any `ref` input | allow zero or one of `--ref-result` / `--ref-operator-file` |
| `run-test-convert` | reject any `ref` input | require exactly one of `--ref-result` / `--ref-operator-file` |
| `run-test-optimize` | reject any `ref` input | require exactly one of `--ref-result` / `--ref-operator-file` |

Implications:

- `run-test-convert --test-mode standalone --ref-result ...` must fail with `parser.error(...)`.
- `run-test-convert --test-mode differential` with no reference input must fail with `parser.error(...)`.
- `run-test-optimize` keeps its current strict differential requirement and keeps the existing standalone rejection of reference inputs.
- `run-test-baseline` keeps its current differential ability to run without a reference input so it can generate baseline result payloads for later reuse.

The standalone rejection is not a new contract for baseline or optimize. The current helper already rejects `--ref-result` and `--ref-operator-file` whenever the resolved test mode is not `differential`. This design keeps that behavior and makes `run-test-convert` follow the same rule from day one.

### Reference Input Semantics

`run-test-convert` and `run-test-optimize` should continue supporting both existing differential reference shapes:

- `--ref-result <archived-result.pt>`
- `--ref-operator-file <reference-operator.py>`

If `--ref-operator-file` is provided:

- derive the expected archived result path with the existing `<stem>_result.pt` rule beside the reference operator
- reuse that payload when it already exists
- otherwise execute the reference operator first with the same test file and resolved mode, then compare against the newly produced archived result

This keeps convert and optimize flows aligned with the current archived-result reuse behavior while changing only the command naming and argument validation contract.

## MCP Behavior

Expose a new managed MCP tool named `run-test-convert`.

The MCP run-test tools should not all mirror the helper CLI in the same way. Keep the tool surfaces workflow-specific:

- `run-test-baseline` remains intentionally narrower than the helper CLI and should continue not to expose `ref_result` or `ref_operator_file`.
- `run-test-convert` must expose both `ref_result` and `ref_operator_file`, because differential convert validation requires exactly one of them.
- `run-test-optimize` continues exposing `ref_result` and `ref_operator_file` for the same reason.

Because `run-test-baseline` stays narrow, its MCP tool description should also stay narrow. Do not describe it as accepting optional archived-result comparison inputs that the tool schema does not expose.

The new `run-test-convert` tool should:

- accept `test_file`, `operator_file`, `ref_result`, `ref_operator_file`, `test_mode`, `remote`, and `remote_workdir`
- describe the workflow in convert terms rather than optimize terms
- invoke the staged helper with `run-test-convert`
- hide the same internal-only parameters that the existing run-test MCP tools already hide

`run-test-baseline` and `run-test-optimize` remain available. This change is additive, not a breaking MCP rename.

## Execution Semantics

### Shared Runtime

`run-test-convert` should reuse the existing run-test execution path:

- shared `_add_run_test_arguments(...)`
- shared local and remote test runners
- shared archived-result comparison helpers
- shared reference-result derivation and auto-production flow

This change should not introduce a fourth test runner implementation.

### Convert-Specific Boundaries

`run-test-convert` should behave like convert validation, not optimize-round bookkeeping:

- expand the top-level run-test dispatch gate from `{run-test-baseline, run-test-optimize}` to include `run-test-convert`
- add `run-test-convert` to `_guard_operator_execution_env(...)` so it gets the same operator-execution environment protection as the other run-test helper commands
- do not attach optimize round timing context
- do not emit optimize-round timing events
- do not inherit optimize-only immediate cleanup side effects that exist specifically for `run-test-optimize`

The optimize-only timing and cleanup behavior should remain scoped to `run-test-optimize`.

## Implementation Shape

Keep the validation logic centralized in the skill-local helper CLI instead of scattering special cases through multiple call sites.

There are only two validation contracts in this change:

- the baseline contract used only by `run-test-baseline`
- the strict contract shared by `run-test-convert` and `run-test-optimize`

The implementation should reflect that small shape. A lightweight helper keyed by command name is acceptable, but a broad new profile layer is unnecessary. The important requirement is that `_validate_run_test_comparison_inputs(...)` stops hardcoding optimize-only error strings and emits command-specific messages such as `run-test-convert differential mode requires exactly one of --ref-result or --ref-operator-file`.

## Documentation Changes

Update the run-eval guidance so the public run-test family becomes:

- `run-test-baseline`
- `run-test-convert`
- `run-test-optimize`

Update both convert skills:

- `skills/triton/triton-npu-convert-pytorch-operator/SKILL.md`
- `skills/tilelang/tilelang-npu-convert-pytorch-operator/SKILL.md`

Those skills should:

- use `run-test-convert` for standalone validation examples
- use `run-test-convert` for differential validation examples
- prefer `--ref-operator-file <original>` as the documented differential convert example
- keep compatibility wording accurate by noting that `--baseline-operator-file` remains an accepted alias where relevant

Update the common run-eval docs:

- `skills/common/ascend-npu-run-eval/SKILL.md`
- `skills/common/ascend-npu-run-eval/references/run-test.md`

These docs should state the stricter option rules explicitly so staged agents are guided toward valid command shapes.
In particular, the `SKILL.md` command index line that currently lists ``run-test-baseline` / `run-test-optimize`` must be expanded to include `run-test-convert`.

## Files

| File | Change |
| --- | --- |
| `docs/specs/2026-07-07-run-test-convert-command-design.md` | Record the convert-specific run-test command contract |
| `skills/common/ascend-npu-run-eval/scripts/cli.py` | Add `run-test-convert` and command-aware reference validation |
| `skills/common/ascend-npu-run-eval/SKILL.md` | Document the three-command run-test surface |
| `skills/common/ascend-npu-run-eval/references/run-test.md` | Add convert guidance and strict mode/reference rules |
| `skills/triton/triton-npu-convert-pytorch-operator/SKILL.md` | Switch convert validation guidance to `run-test-convert` |
| `skills/tilelang/tilelang-npu-convert-pytorch-operator/SKILL.md` | Switch convert validation guidance to `run-test-convert` |
| `src/triton_agent/eval/mcp_server.py` | Expose the new MCP tool and route it to the staged helper |
| `tests/test_skill_command_script.py` | Cover parser, validation, and dispatch behavior for `run-test-convert` |
| `tests/test_run_eval_mcp_server.py` | Cover tool registration and argument forwarding |
| `tests/test_run_eval_mcp_server_tool_metadata.py` | Lock the new MCP tool schema, baseline-vs-convert/optimize parameter differences, and hidden-parameter behavior |
| `tests/test_generation_contracts.py` | Lock updated convert and run-eval skill wording, including the convert-skill `run-test-convert` assertions and the run-eval router command list |

## Verification

- Run focused helper-script tests for parser, validation, and dispatch behavior.
- Run focused MCP server and tool-metadata tests for the new tool.
- Run wording/contract tests that cover run-eval and convert skill docs.
- Run `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/cli.py`.
