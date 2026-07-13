# Convert Batch And Package Split Design

## Summary

- Rename the single-workspace conversion command from `gen-convert` to `convert`.
- Add a new `convert-batch` command that runs the convert workflow across multiple operator workspaces.
- Move convert-specific runtime code out of `generation/` and into a dedicated `convert/` package.
- Keep `generation/` focused on test, benchmark, and eval artifact generation.
- Remove `gen-convert` entirely instead of keeping a compatibility alias.

## Problem

The current convert workflow has two ownership problems.

First, the user-facing command name is misleading. `gen-convert` is no longer just another artifact-generation command. It converts one PyTorch operator into a Triton NPU-backed operator and validates correctness through differential testing. That is a distinct workflow, not just another variation of test or benchmark generation.

Second, the runtime code still lives under `generation/`. That package is a good home for:

- `gen-test`
- `gen-bench`
- `gen-eval`
- `gen-eval-batch`

It is not a good long-term home for convert behavior, especially once batch conversion is added.

If we add a batch command but leave convert under `generation/`, we preserve a misleading boundary and make future convert features harder to place cleanly.

## Goals

- Rename the convert command to `convert`.
- Add `convert-batch` with predictable batch behavior aligned with existing batch commands.
- Give convert its own package with clear ownership for single-workspace and batch conversion orchestration.
- Remove `gen-convert` without a compatibility alias.
- Keep convert semantics unchanged from the current differential-only design:
  - convert one operator
  - preserve the trailing input-helper block
  - validate through differential testing against the original operator

## Non-Goals

- Do not change the convert workflow back into a benchmark or baseline workflow.
- Do not change optimize ownership or optimize batch behavior.
- Do not redesign the differential-test contract itself.
- Do not add a convert-specific new skill; continue using the existing convert skill contract.
- Do not preserve `gen-convert` as a deprecated alias.

## User-Facing Command Contract

### Rename

The public CLI should expose:

- `convert`
- `convert-batch`

The CLI should no longer expose:

- `gen-convert`
- `gen_convert`

This is a real command rename, not a soft alias migration.

### `convert`

`convert` should preserve the current single-workspace convert semantics:

- read one original PyTorch operator file
- write one converted Triton NPU-backed operator file
- generate and execute a differential test
- treat the original operator as the correctness oracle

The main user-visible change for this command is its name and its new internal ownership boundary.

### `convert-batch`

`convert-batch` should run one logical `convert` request per discovered workspace.

It should support:

- `--input` pointing at a root directory of operator workspaces
- `--input` pointing directly at one operator workspace directory
- `--agent`
- `--remote`
- `--remote-workdir`
- `--max-concurrency`
- `--show-output`
- `--verbose`
- `--test-mode differential`

It should not support:

- `--output`
- `--interact`
- benchmark-related options

## Package Boundary

Create a new package:

- `src/helix/convert/`

Its responsibilities should include:

- single-workspace convert request construction
- convert runner invocation
- convert output-path handling
- batch convert workspace discovery and execution

This package should become the owner of convert runtime behavior instead of `generation/`.

### Proposed Modules

- `src/helix/convert/orchestration.py`
  - single-workspace convert request building
  - convert staged-skill allowlist
  - convert runner invocation
- `src/helix/convert/batch.py`
  - `convert-batch` workspace discovery
  - concurrent convert execution
  - summary rendering
- `src/helix/convert/outputs.py`
  - convert output path resolution
  - overwrite checks and cleanup for converted artifacts

If convert-specific dataclasses become useful, add:

- `src/helix/convert/models.py`

Otherwise, keep shared option models only where the boundary remains genuinely shared.

## Generation Boundary After The Split

After this change, `generation/` should no longer own convert behavior.

That means removing convert-specific handling from:

- `src/helix/generation/orchestration.py`
- `src/helix/generation/outputs.py`
- any convert-specific code paths in generation-facing command handlers

`generation/` should remain the home for generation-only workflows:

- test generation
- benchmark generation
- eval generation
- eval batch generation

## Command Routing

Introduce a dedicated command module:

- `src/helix/commands/convert.py`

It should own:

- `handle_convert`
- `handle_convert_batch`

`src/helix/commands/generation.py` should stop routing convert requests.

## Batch Workspace Semantics

`convert-batch` should align with the existing batch UX used elsewhere in the CLI.

### Workspace Discovery

Support both:

1. root directory of workspaces
2. one workspace directory directly

This should match the more ergonomic batch behavior already available in optimize-oriented commands.

### Candidate Selection

The convert batch candidate filter should exclude generated and non-entrypoint files such as:

- `test_*.py`
- `differential_test_*.py`
- `bench_*.py`
- `opt_*.py`
- `triton_*.py`
- `__init__.py`

This avoids treating existing converted artifacts or generated harnesses as fresh source operators.

### Result Rendering

Render one compact line per workspace:

- success: `[OK] <workspace>: converted <operator>.py`
- failure: `[FAIL] <workspace>: <message>`

Then print:

- `Summary: <N> succeeded, <M> failed`

When `--show-output` is enabled, preserve prefixed streaming output with workspace-name attribution.

## CLI Contract Changes

### Command Kinds

Replace the old convert command kind with explicit convert command kinds:

- `CommandKind.CONVERT`
- `CommandKind.CONVERT_BATCH`

Remove:

- `CommandKind.GEN_CONVERT`

### Parser And Help

Update the top-level parser so that:

- help output lists `convert` and `convert-batch`
- command examples use `convert`
- alias normalization no longer maps `gen_convert`

### Output Naming

Single-workspace convert should continue to default to:

- `triton_<stem>.py`

That behavior should move under convert-owned output helpers instead of remaining in generation-owned helpers.

## Prompt And Skill Contract

The convert prompt and staged skills should keep the current differential-only contract:

- original operator is source material and correctness oracle
- converted operator is the system under test
- no benchmark generation
- no baseline creation

No new convert skill is required. The existing `triton-npu-convert-pytorch-operator` skill remains the workflow contract.

## Documentation Impact

Update user-facing documentation so it consistently says:

- `convert` instead of `gen-convert`
- `convert-batch` as the batch counterpart
- convert lives conceptually beside optimize as a top-level workflow, not inside generation

This includes:

- command map
- quick start examples
- convert workflow section
- batch workflow section
- relevant design and plan documents that currently mention `gen-convert`

## Testing Impact

Tests should cover:

- parser and help output for `convert` and `convert-batch`
- removal of `gen-convert`
- removal of the `gen_convert` alias
- single-workspace convert handler routing through the new convert command module
- convert batch workspace discovery
- convert batch candidate exclusion for `triton_*.py`
- prefixed streaming output for `convert-batch`
- convert staged-skill ownership moving out of `generation/`
- README and contract tests using the renamed commands

## Migration Notes

This is both a user-facing rename and an internal ownership correction.

Implementation should therefore treat it as one coherent change:

- rename the command surface
- move convert runtime code into `convert/`
- add batch convert in the new package
- delete the old generation-owned convert routing

Doing these separately would leave temporary mixed ownership and make tests noisier than necessary.

## Open Questions Resolved

- The single-workspace command should be renamed to `convert`.
- The batch command should be named `convert-batch`.
- `gen-convert` should not be kept as an alias.
- Convert should move into its own package instead of staying under `generation/`.
