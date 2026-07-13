# Optimize Workflow State Resume Bootstrap Design

## Goal

Let optimize sessions resume cleanly across restarts even though
`.helix/` remains a temporary runtime directory that is cleaned up after
each session.

## User-Visible Semantics

- All optimize entry points that currently manage workflow state should share the
  same startup behavior:
  - reuse an existing valid `.helix/state.json` when present
  - otherwise, if durable optimize artifacts prove the workspace is resumable,
    rebuild a minimal `.helix/state.json`
  - otherwise bootstrap a fresh baseline-phase workflow state
- Rebuilt workflow state must not automatically reopen an active round.
- When state is rebuilt from durable artifacts, the restored state should be:
  - `phase=awaiting_round_start`
  - `baseline.status=passed`
  - `baseline.submitted_at=null`
  - `current_round=null`
- After this restore, the agent must run `ascend-npu-optimize-state`
  `start-round` before editing or submitting a new `opt-round-N/`.
- `.helix/` should still be removed at normal session cleanup so users do
  not need to know about or manually manage that runtime directory.

## Problem

The repository currently has two incompatible optimize bootstrap behaviors:

- runner-managed optimize sessions create `.helix/state.json` at startup
  and remove `.helix/` at cleanup
- Claude plugin optimize sessions currently create `.helix/` only and, if
  `state.json` is missing, return repair guidance instead of rebuilding state

That behavior breaks restartable optimize workflows. If a workspace already has
durable optimize artifacts such as:

- `baseline/`
- `opt-note.md`
- generated test and benchmark harnesses
- `opt-round-*`

then a later optimize launch should continue from that durable state. Today it
cannot, because the runtime-owned `.helix/state.json` was cleaned up and
there is no shared reconstruction path.

## Design

### Shared Bootstrap Helper

Add one shared optimize bootstrap helper under `src/helix/optimize/`.

It should own these startup branches:

1. If `.helix/state.json` exists and is valid, reuse it unchanged.
2. If `state.json` is missing, inspect durable optimize workspace artifacts
   using the existing resumable-session checks instead of ad-hoc filesystem
   guessing.
3. If the workspace is resumable, rebuild `.helix/state.json` as a
   minimal awaiting-round-start state.
4. If the workspace is not resumable, bootstrap a fresh baseline-phase state.

The shared helper must support two caller shapes:

- runner-managed startup, which already knows the requested source operator path
- Claude plugin `SessionStart`, which does not receive a source operator path in
  hook payloads

Implementation choice:

- accept an optional source-operator hint from the caller
- when no hint is provided, resolve the operator from `baseline/state.json`
  before running resumable-session classification
- if no hint is provided and no durable baseline exists yet, treat the workspace
  as a fresh baseline bootstrap candidate

This helper should become the single source of truth for both:

- runner-managed optimize startup
- Claude plugin `SessionStart`

Implementation choice:

- keep the durable recovery algorithm in `src/helix/optimize/`
- keep `hooks/claude_plugin/state_bootstrap.py` as a thin plugin-local wrapper
  that imports and delegates to that shared helper

The Claude plugin build already copies only plugin files plus bundled skills, so
the hook cannot assume the repository checkout is available by path. Therefore
the plugin build step must also package the minimal Python support needed for
that shared optimize bootstrap helper and make it importable to the plugin-local
hook scripts. Do not re-implement the recovery algorithm separately inside
`hooks/claude_plugin/state_bootstrap.py`.

### Restore Source Of Truth

Reuse the existing optimize workspace classification logic in
`src/helix/optimize/resume.py` to decide whether durable artifacts
describe a resumable optimize session.

Do not create a second independent “artifact recovery” scanner.

This keeps recovery aligned with the same baseline and harness contracts already
used by optimize resume behavior elsewhere in the CLI.

The durable source of truth for the original operator path remains
`baseline/state.json`. The temporary runtime workflow state must not duplicate
that field.

### Runtime Directory Ordering And Crash Residue

The shared helper should run after the caller has claimed or prepared the
workspace-local `.helix/` directory, not before.

For runner-managed optimize startup:

- keep `_prepare_hidden_helix_dir(...)` as the owner of runtime-dir
  creation and stale-runtime rejection
- let it continue failing fast when `.helix/` already exists with
  unexpected content
- call the shared helper only after that directory is available

This preserves the current “stale hidden runtime dir is an explicit problem”
contract for crashed or interrupted runner-managed sessions.

For Claude plugin startup:

- keep the existing plugin-local best-effort creation path for `.helix/`
- if the directory already exists, inspect `state.json` through the shared
  helper logic rather than deleting the directory silently

Do not auto-delete a pre-existing `.helix/` directory in either path.
Unexpected runtime residue should remain visible as a recoverable or explicit
failure condition, not be silently discarded.

### Restored State Shape

When restoring from durable artifacts:

- treat baseline as already established
- do not infer the previous active round as still active
- do not infer mid-round strategy state
- require an explicit new `start-round` command before any round-local edits

This intentionally trades a little extra ceremony for predictable semantics and
avoids guessing whether an unfinished `opt-round-N/` should still be considered
live.

`baseline.submitted_at=null` is intentional for rebuilt state. It means:

- baseline is considered reusable for phase progression
- this state was reconstructed from durable artifacts rather than advanced by a
  fresh `submit-baseline` event in the current session

This is the same observability distinction already used by the existing
runner-managed `baseline_reused=True` bootstrap path.

### Runtime Workflow State Scope

`.helix/state.json` should contain only temporary workflow-tracking data
that is not already persisted elsewhere. In particular:

- keep fields such as `schema_version`, `run_id`, `phase`, `baseline`, `rounds`,
  and `current_round`
- remove `source_operator` from runtime workflow state
- continue treating `baseline/state.json` as the durable source for
  `source_operator`, generated harness paths, and other baseline metadata

This avoids redundant state drift and keeps runtime rebuild semantics simple:
recreate temporary phase tracking from durable artifacts instead of copying
durable metadata into a second file.

### Baseline-Phase Edit Policy

Built-in edit policy should no longer depend on `source_operator` during the
baseline phase.

Updated semantics:

- `baseline` phase may allow ordinary in-workspace built-in edits needed to
  establish or repair the baseline
- `awaiting_round_start` still blocks round edits until `start-round`
- `round_active` still restricts built-in edits to the active `opt-round-N/`
  plus approved top-level progress files
- protected internal paths remain denied in every phase, including:
  - `.helix/`
  - `helix-logs/`
  - backend-managed staged hook and skill implementation directories

This intentionally relaxes baseline gating because precise source-operator
tracking is already available from durable baseline metadata when needed, and
baseline should not require path-exact interception to be usable.

### Invalid Existing `state.json`

If `.helix/state.json` exists but is malformed or fails schema
validation:

- do not silently delete it
- do not fall through to durable-artifact rebuild in the same startup attempt

Instead:

- runner-managed optimize should fail explicitly with a message telling the user
  to remove `.helix/` or restart from a clean runtime state
- Claude plugin hooks should surface repair guidance and deny optimize editing
  until the invalid runtime state is repaired or removed

This keeps corrupted hidden state from being silently ignored while still
allowing “state missing but durable artifacts valid” to be auto-recovered.

### Cleanup Behavior

Do not change cleanup ownership:

- runner-managed cleanup still removes `.helix/`
- Claude plugin `SessionEnd` still removes `.helix/`

The resumability contract should come from durable optimize artifacts plus
shared bootstrap logic, not from keeping hidden runtime state around between
sessions.

### Preflight Re-Bootstrap Removal

The existing optimize execution path contains a second bootstrap call after
baseline preflight succeeds for an already-reusable baseline.

That post-preflight rewrite should be removed once startup bootstrap is handled
by the shared helper. After this change:

- startup bootstrap decides fresh-baseline vs reusable-baseline state exactly
  once
- successful baseline preflight should not rewrite `.helix/state.json`
  again just to restate `baseline_reused=True`

This keeps workflow-state creation single-sourced and avoids a confusing second
write path.

### Claude Plugin Alignment

Claude plugin optimize sessions should align to the same startup semantics as
runner-managed optimize sessions:

- fresh workspace: create a baseline-phase workflow state
- resumable workspace with no `state.json`: rebuild awaiting-round-start state
- valid existing `state.json`: reuse it

This replaces the current plugin-only “missing state means repair manually”
behavior.

Because Claude hook payloads do not include the current source operator path,
the shared helper must be able to rebuild resumable state from durable baseline
metadata alone. The plugin wrapper should stay thin and delegate that decision
to the shared helper instead of implementing its own recovery rules.

### `run_id` Semantics

When state is rebuilt from durable artifacts, use the current session's newly
allocated archive `run_id`, not the prior session's run id.

This is acceptable because:

- `.helix/state.json` is runtime-scoped rather than durable history
- per-session archive output already lives under a fresh
  `helix-logs/<run-id>/`
- rebuild is starting a new optimize session that resumes durable workflow
  artifacts, not reopening the old session archive namespace

Downstream consumers should treat rebuilt state as “current session resumes old
workspace artifacts” rather than “current session is the same archived run.”

## Non-Goals

- Do not auto-restore `phase=round_active`.
- Do not infer or recreate previous round strategy-state fields from old
  `opt-round-N/` artifacts.
- Do not keep `.helix/` after normal session cleanup.
- Do not change durable optimize artifact contracts such as `baseline/state.json`
  or `opt-round-N/round-state.json`.

## Validation

Add focused coverage for:

- shared bootstrap helper: existing valid state, resumable-artifact restore, and
  fresh baseline bootstrap
- shared bootstrap helper failure paths:
  - invalid existing `.helix/state.json`
  - partial optimize session detected by resume classification
  - corrupted or missing `baseline/state.json` inside an otherwise
    optimize-looking workspace
- runner-managed optimize artifact preparation: restored state is created when a
  resumable workspace has no `.helix/state.json`
- runner-managed optimize startup still fails fast on unexpected pre-existing
  `.helix/` directory contents
- Claude plugin hooks: `SessionStart` rebuilds awaiting-round-start state from
  resumable durable artifacts and bootstraps fresh baseline state for new
  workspaces
- cleanup behavior remains unchanged: `.helix/` is still removed at
  session end

Suggested test locations:

- shared helper and runner-managed startup:
  - `tests/test_optimize_workflow_state.py`
  - `tests/test_optimize_guidance.py`
- Claude plugin startup and hook behavior:
  - `tests/test_claude_optimize_plugin_hooks.py`
- plugin packaging changes, if new shared runtime support files must be copied:
  - `tests/test_claude_optimize_plugin.py`
- baseline-phase guard changes:
  - `tests/test_codex_pretooluse_guard.py`
  - `tests/test_opencode_hook_guard.py`

The test updates should explicitly replace current expectations that:

- Claude plugin `SessionStart` creates `.helix/` without `state.json`
- Claude plugin always returns repair guidance when `state.json` is absent
- optimize execution rewrites workflow state after baseline preflight via the
  old post-preflight `baseline_reused=True` bootstrap path
- runtime workflow state includes `source_operator`
- baseline-phase built-in edit allowlists depend on matching the source operator
