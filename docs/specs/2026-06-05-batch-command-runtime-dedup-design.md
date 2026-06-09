# Batch Command Runtime Dedup Design

## Summary

- Deduplicate the shared batch orchestration used by `gen-eval-batch`, `convert-batch`, and `optimize-batch`.
- Align all three commands on one shared batch result model with `ok`, `failed`, and `skipped` states.
- Extend `gen-eval-batch` and `convert-batch` with root-level batch status files so interrupted runs can skip already completed workspaces on rerun.
- Upgrade `optimize-batch` status tracking from operator-name-only matching to command-specific configuration fingerprints.
- Keep optimize-only side effects such as auto-upload and auto-report outside the shared batch runtime.

## Problem

The current batch command implementations already share the same execution shape:

- discover workspaces under one root
- resolve one operator file per workspace
- run one request per workspace in a thread pool
- optionally prefix interleaved `--show-output` lines
- optionally wrap the whole batch in one managed MCP scope
- optionally allocate per-workspace NPU affinity
- collect per-workspace results and print a summary

That shape currently lives in three separate modules:

- `src/triton_agent/generation/batch.py`
- `src/triton_agent/convert/batch.py`
- `src/triton_agent/optimize/batch.py`

This duplication has already created drift:

- `gen-eval-batch` and `convert-batch` still use their own result dataclasses and summary renderers.
- `optimize-batch` has richer `ok | failed | skipped` semantics while the other two still model success as a boolean.
- only `optimize-batch` can skip already completed workspaces after interruption
- status-file matching in `optimize-batch` only records the selected operator filename, so important semantic changes such as prompt, runtime mode, or MCP toggles can be skipped incorrectly

The repository now has enough batch commands that repeated fixes to concurrency, output streaming, MCP handling, and resumable batch status will keep spreading unless the batch runtime is centralized.

## Goals

- Remove the repeated executor, output-stream, MCP-scope, and NPU-affinity code from the three main batch command modules.
- Keep command-specific logic local to each command instead of hiding it behind one large generic options type.
- Align `gen-eval-batch`, `convert-batch`, and `optimize-batch` on one result model and one summary renderer.
- Give all three commands resumable batch completion tracking through visible root-level `*-batch-status.json` files.
- Make completed-workspace skipping depend on both the selected operator filename and a command-specific configuration fingerprint.
- Preserve current user-facing command names and single-workspace behavior.

## Non-Goals

- Do not change single-workspace `gen-eval`, `convert`, or `optimize` command semantics.
- Do not move optimize-specific upload or report behavior into shared batch infrastructure.
- Do not unify `verify-batch` and `log-check-batch` into the new shared executor in this change.
- Do not make batch status depend on code-agent backend choice, verbosity, `--show-output`, or concurrency.
- Do not fingerprint external mutable state such as installed agent versions, repository code revisions, or compiler source checkout contents.

## Current Shared Shape And True Differences

The three batch commands share the following orchestration behavior today:

- workspace discovery with root-directory fallback
- hidden-directory filtering
- thread-pool based workspace execution
- `PrefixedTextStream` for `--show-output`
- `managed_mcp_scope()` wrapping when the staged skill set requires MCP servers
- `TRITON_AGENT_BATCH_NPU_DEVICES` and `TRITON_AGENT_BATCH_WORKERS_PER_NPU` handling
- defensive conversion of unexpected worker exceptions into per-workspace failures

The real command-specific differences are narrower:

### Operator selection rules

- `gen-eval-batch` excludes:
  - `test_*`
  - `differential_test_*`
  - `bench_*`
  - `opt_*`
  - `__init__.py`
- `convert-batch` excludes the same names and also excludes:
  - `triton_*`
- `optimize-batch` currently uses the same exclusion set as `gen-eval-batch`

### Single-workspace request construction

- `gen-eval-batch` builds requests with `build_generation_request(...)`
- `convert-batch` builds requests with `build_convert_request(...)`
- `optimize-batch` builds requests with `build_optimize_request(...)`

### Batch-local follow-up behavior

- `optimize-batch` writes a status file, skips matching completed workspaces, and optionally runs auto-upload and auto-report after success
- `gen-eval-batch` and `convert-batch` currently have no batch-local persistent state

### Result wording

- success messages differ by command
- failure fallback text differs by command
- unexpected exception labels differ by command

This means the right abstraction is a shared runtime skeleton with command-local adapters, not a fully generic workflow layer.

## User-Facing Behavior After Refactor

### Shared result semantics

`gen-eval-batch`, `convert-batch`, and `optimize-batch` should all report:

- `[OK] <workspace>: <message>`
- `[FAIL] <workspace>: <message>`
- `[SKIP] <workspace>: <message>`

The final summary for all three commands should be:

```text
Summary: <N> succeeded, <M> failed, <K> skipped
```

For `gen-eval-batch` and `convert-batch`, `skipped` will become a normal state rather than an optimize-only special case.

### Batch status files

Each command should own one visible root-level status file:

- `gen-eval-batch-status.json`
- `convert-batch-status.json`
- `optimize-batch-status.json`

Each file records completion by workspace relative path under the batch root.

### Skip behavior

A workspace is skipped only when all of the following are true:

- the status file parses successfully
- the workspace has an entry
- the entry status is `completed`
- the recorded operator filename still matches the operator selected for this run
- the recorded configuration fingerprint matches the current command fingerprint

Any mismatch means the workspace is runnable again.

### Reset behavior

- `optimize-batch --reset-optimize` should continue clearing `optimize-batch-status.json` before scheduling workspaces.
- `gen-eval-batch` and `convert-batch` do not gain new reset flags in this change.
- For `gen-eval-batch` and `convert-batch`, users can delete the visible status file manually when they want a full rerun with unchanged options.

## Shared Batch Status Schema

The three status files should use one common schema and one shared reader/writer implementation.

Shape:

```json
{
  "version": 2,
  "workspaces": {
    "matmul": {
      "status": "completed",
      "operator_file": "kernel.py",
      "config_fingerprint": "sha256:..."
    },
    "layernorm": {
      "status": "incomplete",
      "operator_file": "op.py",
      "config_fingerprint": "sha256:..."
    }
  }
}
```

Rules:

- `version` is the shared batch-status schema version, not a per-command feature counter.
- `workspaces` keys are workspace paths relative to the batch root.
- the root workspace still uses `"."` when `--input` points directly at one operator workspace directory.
- each entry stores:
  - `status`: `completed` or `incomplete`
  - `operator_file`: selected operator filename
  - `config_fingerprint`: canonical hash of the relevant command options
- unknown top-level keys are ignored
- unknown workspace fields are ignored
- malformed files or unsupported versions are treated as if the file were missing

### Versioning and migration

`optimize-batch` already writes a version-1 file that does not contain `config_fingerprint`.

After this change:

- `optimize-batch` writes `version: 2`
- existing `version: 1` `optimize-batch-status.json` files are treated as stale and ignored
- the next optimize batch run after upgrade reruns all workspaces once and rewrites the file in the new format

`gen-eval-batch` and `convert-batch` start directly on the shared version-2 schema.

## Configuration Fingerprints

Completed-workspace reuse must be invalidated when options that change batch semantics or produced artifacts change.

The shared status helper should compute fingerprints from:

1. a command-local canonical payload
2. stable JSON serialization with sorted keys and normalized `null`
3. a SHA-256 hex digest stored as `sha256:<digest>`

The shared helper owns the hashing mechanism.
Each command adapter owns the canonical payload fields.

### Fields that should not be fingerprinted

These options should not invalidate completed workspaces:

- `agent_name`
- `verbose`
- `show_output`
- `log_tools`
- `max_concurrency`
- `remote_workdir`
- optimize auto-upload and auto-report toggles
- optimize subagent and agent-hook execution toggles

These are execution or observability knobs, not output/behavior contracts.

### `gen-eval-batch` fingerprint payload

Include:

- `test_mode`
- `bench_mode`
- `prompt`
- `remote`
- `enable_mcp`

Do not include:

- `force_overwrite`
- `agent_name`
- `verbose`
- `show_output`
- `log_tools`

### `convert-batch` fingerprint payload

Include:

- `test_mode`
- `prompt`
- `remote`
- `enable_mcp`

Do not include:

- `force_overwrite`
- `agent_name`
- `verbose`
- `show_output`
- `log_tools`

### `optimize-batch` fingerprint payload

Include:

- `min_rounds`
- `resume_mode`
- `round_mode`
- `round_batch_size`
- `test_mode`
- `bench_mode`
- `prompt`
- `remote`
- `target_chip`
- `optimize_target`
- `optimize_knowledge`
- `compiler_source_analysis`
- `enable_cann_ext_api`
- `enable_mcp`

Do not include:

- `agent_name`
- `verbose`
- `show_output`
- `log_tools`
- `upload_enabled`
- `report`
- `enable_subagent`
- `enable_agent_hooks`
- `no_agent_session`

The excluded optimize fields change how the command is executed or observed, but not the intended completed workspace contract.

## Shared Runtime Design

Add a shared batch runtime module, for example:

- `src/triton_agent/batch_runtime.py`

Its job is to own the repeated orchestration mechanics only:

- workspace discovery and root fallback
- thread-pool execution
- prefixed `--show-output` stream routing
- NPU-affinity validation and slot assignment
- one managed MCP scope around the full batch
- per-workspace result collection
- defensive conversion of unexpected exceptions into failed results

It should not own command semantics, prompt building, or optimize-only follow-up behavior.

### Shared runtime input shape

Use one adapter/spec object per batch command, for example `BatchCommandSpec`, containing:

- `status_filename`
- `resolve_operator_file(workspace) -> Path`
- `build_request(item, options) -> AgentRequest`
- `run_request(request, stdout?, stderr?) -> AgentResult`
- `summarize_failure(result) -> str`
- `success_message(item, result) -> str`
- `unexpected_failure_message(exc) -> str`
- `staged_skill_names(options) -> tuple[str, ...]`
- `fingerprint_payload(options) -> dict[str, object]`

Optional hooks:

- `pre_schedule(item, options) -> BatchCommandResult | None`
  - allows command-local validation before submission
- `should_skip_completed(root, item, status_entry, options) -> bool`
  - default behavior can compare operator filename and fingerprint
  - commands can override only if they need narrower rules later
- `after_success(root, item, result, options) -> None`
  - used only for command-local side effects after the shared runtime has already written the completed status entry
- `after_failure(root, item, result_or_exc, options) -> None`
  - used only for command-local failure side effects after the shared runtime has already written the incomplete status entry

The shared runtime should expose a narrow `run_batch_operator_command(...)` entrypoint that returns the final process exit code after rendering results.

### Shared workspace and result types

Create shared dataclasses:

- `BatchOperatorWorkspace`
  - `workspace: Path`
  - `operator_file: Path`
- `BatchCommandResult`
  - `workspace: Path`
  - `status: Literal["ok", "failed", "skipped"]`
  - `message: str`

`BatchCommandResult` replaces:

- `BatchGenEvalResult`
- `BatchConvertResult`

`BatchOptimizeResult` can either become a thin alias or be removed in favor of the shared type.

## Shared Result Rendering

Add one shared renderer, for example:

- `src/triton_agent/batch_results.py`

Responsibilities:

- sort by workspace name
- print `[OK]`, `[FAIL]`, and `[SKIP]`
- print `Summary: X succeeded, Y failed, Z skipped`
- return exit code `0` only when there are results and none failed

`src/triton_agent/optimize/render.py` can remain as a thin compatibility wrapper if that keeps import churn small.

## Command-Local Responsibilities After Refactor

### `gen-eval-batch`

Keep local:

- candidate exclusion rules
- generation request construction
- failure summary wording
- success message wording
- fingerprint payload definition
- status filename constant

Add:

- batch status read/update support through the shared helper
- `skipped` rendering through the shared result model

### `convert-batch`

Keep local:

- convert-specific candidate exclusion rules
- convert request construction
- failure summary wording
- success message wording
- fingerprint payload definition
- status filename constant

Add:

- batch status read/update support through the shared helper
- `skipped` rendering through the shared result model

### `optimize-batch`

Keep local:

- optimize request construction and validation
- optimize status filename constant
- optimize-specific fingerprint payload definition
- `--reset-optimize` status-file clearing
- optional auto-upload
- optional auto-report

Remove from local ownership:

- generic thread-pool orchestration
- generic MCP-scope wrapping
- generic affinity plumbing
- generic batch-result rendering

## Skip And Write Semantics

The shared status helper should follow these rules for all three commands:

### Read time

- missing file: treat all workspaces as runnable
- malformed JSON: treat all workspaces as runnable
- unsupported version: treat all workspaces as runnable
- malformed workspace entry: ignore that one entry and continue

### Success write

After a workspace succeeds, write or update:

- `status: "completed"`
- `operator_file: <selected filename>`
- `config_fingerprint: <current fingerprint>`

### Failure write

After a workspace fails or raises an unexpected exception, write or update:

- `status: "incomplete"`
- `operator_file: <selected filename>` when available
- `config_fingerprint: <current fingerprint>`

### Skip write

Skipped workspaces do not need a write on the current run when the status file already matched.

The runtime may still rewrite the file in memory order if that simplifies implementation, but it should not change the logical meaning of a matched completed entry.

## Documentation And Cleanup Impact

The implementation should update user-facing docs to mention:

- the new `gen-eval-batch-status.json`
- the new `convert-batch-status.json`
- the new version-2 `optimize-batch-status.json` fingerprint behavior
- shared `[OK] / [FAIL] / [SKIP]` output shape for all three batch commands

Cleanup logic should also recognize the new generated batch files:

- `gen-eval-batch-status.json`
- `convert-batch-status.json`

## Verification

Add or update tests for:

- shared batch result rendering with skipped rows
- shared fingerprint hashing stability
- batch status schema parsing and invalid version fallback
- `gen-eval-batch` skipping completed workspaces when fingerprint matches
- `gen-eval-batch` rerunning workspaces when fingerprint changes
- `convert-batch` skipping completed workspaces when fingerprint matches
- `convert-batch` rerunning workspaces when fingerprint changes
- `optimize-batch` ignoring old version-1 status files
- `optimize-batch` rerunning workspaces when fingerprint changes
- `optimize-batch --reset-optimize` still clearing the status file before scheduling
- summary text for all three commands including skipped counts

Repository verification should continue using the standard commands:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
