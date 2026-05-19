# Optimize PT Cleanup Control Design

## Goal

Change optimize PT cleanup semantics so baseline validation never deletes archived `.pt` files, ordinary optimize runs keep those files by default, and an explicit environment variable can re-enable round and end-of-run PT cleanup when desired.

## Current Problem

Today optimize cleanup treats archived correctness payloads as disposable scratch output in multiple places:

- `check-baseline` deletes matching `.pt` files from both `baseline/` and the workspace root after a passing baseline check.
- `check-round` deletes matching `.pt` files from the round directory after a passing round check.
- optimize execution cleanup scans the workspace root and each `opt-round-*` directory and deletes matching `.pt` files after the agent run ends.
- `--reset-optimize` deletes workspace-root `*_result.pt` files as part of its fresh-start cleanup.

That behavior makes it easy to lose archived differential results that are still useful for later inspection or manual comparison, especially after a successful baseline check or a normal optimize run.

## Desired Behavior

- `check-baseline` must never delete `.pt` files.
- Normal optimize execution must preserve matching `.pt` files by default.
- A new environment variable, `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES`, should opt back into the current round and end-of-run cleanup behavior.
- `--reset-optimize` must keep its current behavior and continue deleting workspace-root `*_result.pt` files regardless of the new environment variable.
- Cleanup scope must stay limited to the existing optimize-owned PT naming rules:
  - `test_result.pt`
  - any filename ending with `_result.pt`

## Non-Goals

- Do not change PT cleanup behavior outside optimize.
- Do not broaden cleanup to arbitrary `.pt` files such as `model.pt`.
- Do not add a CLI flag for this behavior.
- Do not change `--reset-optimize` semantics.

## Decision

Use one explicit enable-only environment variable:

```text
TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES
```

Behavior:

- Unset: preserve PT files.
- Case-insensitive values `1`, `true`, `yes`, or `on`: delete PT files in the ordinary optimize cleanup paths.
- Any other value: preserve PT files.

This keeps the default safe, avoids widening the CLI surface, and gives users a simple way to restore the old cleanup behavior for disk-sensitive runs.

## Design

### Single Source Of Truth

Keep optimize PT cleanup policy anchored in the `triton-npu-optimize-check` skill-side contract code and continue reusing that logic from runtime through the existing bridge layer.

This follows the repository rule that skill-side helper code must not import `triton_agent`, while runtime code may reuse skill-side behavior through `skill_loader`.

Concretely:

- extend `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py` with a small helper that answers whether ordinary optimize PT cleanup is enabled
- expose that helper through the existing `optimize_check` bridge module
- reuse it from `src/triton_agent/optimize/pt_cleanup.py`

This avoids duplicating environment-variable parsing in both `skills/` and `src/`.

### Baseline Check

`check_baseline()` should stop deleting PT files entirely.

After a passing baseline gate:

- return a normal passing result
- do not call PT cleanup for `baseline/`
- do not call PT cleanup for the workspace root
- do not emit cleanup-related success messages

This makes baseline validation a pure check instead of a check-plus-mutation step.

### Round Check

`check_round()` should preserve its existing validation behavior and only make PT deletion conditional.

When `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` is enabled:

- keep deleting matching PT files inside the current round directory
- keep the existing informational issue text that reports which PT files were cleaned

When the variable is not enabled:

- skip round PT deletion entirely
- return the same passing result shape, without cleanup messages

Round cleanup should remain scoped to the checked round directory only; it should not reach into `baseline/` or the workspace root.

### End-Of-Run Optimize Cleanup

`cleanup_workspace_pt_files()` currently deletes matching PT files from:

- the workspace root
- each `opt-round-*` directory

Keep that path structure unchanged, but gate the deletion behind the new environment variable:

- enabled: preserve current cleanup behavior and verbose logging
- disabled: perform no PT deletion and report nothing

The optimize execution flow should keep calling the helper during supervised and unsupervised cleanup so the cleanup point remains centralized, but the helper should become a no-op when PT deletion is disabled.

### Reset Optimize

`reset_optimize_workspace()` should stay unchanged for PT handling.

When `--reset-optimize` is active, workspace-root `*_result.pt` files should still be deleted unconditionally before the fresh run begins, regardless of `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES`.

This preserves the existing meaning of a destructive fresh reset.

### Documentation

Update user-facing docs in `README.md` to describe:

- the new `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` environment variable
- the default preservation behavior for normal optimize cleanup
- the special cases:
  - `check-baseline` never deletes PT files
  - `--reset-optimize` still deletes known optimize PT artifacts

If the CLI environment-variable help block lists optimize-related runtime variables, add this variable there too.

## Verification

Add or update coverage in these areas:

- `tests/test_skill_command_script.py`
  - confirm `check-baseline` no longer deletes `baseline/test_result.pt`
- `tests/test_optimize_checks.py`
  - confirm `check-round` preserves round PT files by default
  - confirm `check-round` deletes round PT files when `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` is enabled
- `tests/test_optimize_runtime.py`
  - confirm ordinary optimize PT cleanup preserves workspace-root and round PT files by default
  - confirm ordinary optimize PT cleanup deletes those files when `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` is enabled
  - confirm `reset_optimize_workspace()` still deletes workspace-root `*_result.pt` files regardless of the environment variable
- `tests/test_cli.py`
  - confirm the top-level environment-variable help text mentions `TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES`

## Scope Boundaries

- Do not rename PT artifacts.
- Do not change optimize artifact discovery beyond the existing PT filename rules.
- Do not move cleanup ownership out of optimize.
- Do not change baseline or round validation semantics apart from the PT-file mutation side effect.
