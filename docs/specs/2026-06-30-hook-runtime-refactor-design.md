# Hook Runtime Refactor Design

## Goal

Replace the ad-hoc Python hook/plugin runtime packaging with a single
standalone `hook_runtime` package so staged hooks and the standalone Claude
optimize plugin can share the same runtime logic without importing
`triton_agent` from inside hook code.

## User-Visible Semantics

- Claude, Codex, and the standalone Claude optimize plugin should keep the same
  guard-policy behavior they have today.
- Optimize workflow bootstrap and resume behavior should keep the same semantics
  that were just introduced:
  - fresh workspace bootstraps baseline phase
  - resumable workspace rebuilds awaiting-round-start state
  - valid existing `.triton-agent/state.json` is reused
- Built standalone plugin artifacts must no longer contain a `python_support/`
  tree.
- Hook entry scripts should remain thin wrappers. Users should not see behavior
  drift between source-tree runs, staged hook runs, and built plugin runs.

## Problem

The current code mixes three different ownership models:

- source-tree hook logic under `hooks/shared/`
- optimize runtime logic under `src/triton_agent/optimize/`
- standalone-plugin-only Python copies under `python_support/`

That creates two structural problems:

1. The standalone Claude plugin depends on copied fragments of `triton_agent`
   even though it is built as a separate artifact.
2. Hook-side Python code currently needs special import-path fallbacks and
   packaging exceptions that do not match real ownership.

The result is confusing module boundaries, duplicated packaging rules, and an
unclear answer to “which runtime code actually belongs to hooks?”

## Design

### Package Ownership

Introduce a standalone Python package at:

- source of truth: `src/hook_runtime/`
- staged or built runtime payload: `hooks/hook_runtime/`

`hook_runtime` owns Python runtime code that must execute inside staged hook
directories or built plugin payloads.

This package must not import `triton_agent`.

`triton_agent` may import `hook_runtime` when the CLI wants to reuse the same
runtime logic.

### What Moves Into `hook_runtime`

Move or adapt the following responsibilities into `src/hook_runtime/`:

- shared PreToolUse wrapper support currently in
  `hooks/shared/pretooluse_guard_support.py`
- Python guard-policy decision logic currently in
  `hooks/shared/tool_use_guard_policy.py`
- optimize workflow bootstrap and restore helpers currently rooted in
  `src/triton_agent/optimize/workflow_state.py`
- the minimal durable optimize-state helpers needed by that bootstrap path,
  including:
  - baseline state loading
  - resumable-workspace inspection
  - skill-script loading for `ascend-npu-optimize-state`

This package should expose small focused modules instead of one large mixed
runtime file. Expected responsibilities are:

- `hook_runtime.pretooluse_adapter`
- `hook_runtime.tool_use_decision`
- `hook_runtime.optimize.workflow_state`
- `hook_runtime.optimize.resume`
- `hook_runtime.optimize.baseline`
- `hook_runtime.skill_loader`

The exact filenames may vary, but the package should preserve these ownership
boundaries.

### Skill Resolution Inside `hook_runtime`

`hook_runtime` must resolve bundled skills without importing
`triton_agent.resources` or `triton_agent.skill_catalog`.

The runtime root should be inferred from its own file location:

- in source-tree mode, `src/hook_runtime/...` resolves the repo root
- in staged or built mode, `hooks/hook_runtime/...` resolves the staged plugin
  root

In both layouts, the runtime should find skills relative to that root:

- source tree: `<root>/skills/...`
- built or staged payload: `<root>/skills/...`

This makes `hook_runtime` self-sufficient and removes the need for
plugin-specific fallback logic in `triton_agent.resources` and
`triton_agent.skill_catalog`.

### Wrapper Structure

Keep these wrapper files as entrypoints only:

- `hooks/claude/pretooluse_guard.py`
- `hooks/codex/pretooluse_guard.py`
- `hooks/claude_plugin/pretooluse_guard.py`
- `hooks/claude_plugin/session_start.py`
- `hooks/claude_plugin/session_end.py`
- `hooks/claude_plugin/state_bootstrap.py`

Their job is only to:

- add the correct import root to `sys.path`
- parse stdin or CLI arguments
- apply host-specific gating
- delegate to `hook_runtime`

They should no longer load sibling `pretooluse_guard_support.py` or
`tool_use_guard_policy.py` files directly.

### Import Bootstrap Rules

Wrapper import bootstrapping should become simple and consistent:

- source-tree wrapper execution should add `<repo>/src` to `sys.path`
- staged or built wrapper execution should add the current `hooks/` directory
  to `sys.path`

That allows wrappers to import:

- `hook_runtime.*` from `src/hook_runtime/` during source-tree execution
- `hook_runtime.*` from `hooks/hook_runtime/` during staged or built execution

Do not keep `python_support` or `shared`-directory import fallbacks after this
refactor.

### Staging And Build Layout

Hook staging and plugin build steps should copy `src/hook_runtime/` into the
runtime payload as `hooks/hook_runtime/`.

Specifically:

- backend hook staging should stage:
  - wrapper entry scripts
  - `hook_runtime/`
  - policy or settings files as before
- Claude optimize plugin build should package:
  - `hooks/claude_plugin/*`
  - `hooks/hook_runtime/`
  - selected `skills/`
  - plugin metadata

It should not package:

- `python_support/`
- copied `triton_agent` fragments for hook use

### CLI Reuse

The CLI should stop treating optimize workflow-state bootstrap as
`triton_agent`-owned hook-private logic.

Instead:

- `src/triton_agent/optimize/workflow_state.py` becomes a thin compatibility
  facade or is reduced to direct delegation into `hook_runtime`
- other CLI modules that need the same behavior may import `hook_runtime`
  directly when that improves boundary clarity

The important boundary is one-way dependency:

- `triton_agent` -> `hook_runtime` is allowed
- `hook_runtime` -> `triton_agent` is not allowed

### Out Of Scope

This refactor does not need to unify the JavaScript hook guard in
`hooks/opencode/triton-agent-hook-guard.js` with Python runtime modules.

That file may keep its current implementation for now.

## File-Level Impact

### New

- `src/hook_runtime/`
- `src/hook_runtime/optimize/`
- tests that target `hook_runtime` directly

### Simplified

- `hooks/claude/pretooluse_guard.py`
- `hooks/codex/pretooluse_guard.py`
- `hooks/claude_plugin/pretooluse_guard.py`
- `hooks/claude_plugin/state_bootstrap.py`
- `scripts/build-claude-optimize-plugin.py`
- `src/triton_agent/backends/claude_hooks.py`
- `src/triton_agent/backends/codex_hooks.py`

### Deleted Or Removed From Runtime Contract

- `python_support/`
- hook-runtime fallbacks in `src/triton_agent/resources.py`
- staged-flat fallback in `src/triton_agent/skill_catalog.py`
- direct staged copies of `hooks/shared/pretooluse_guard_support.py`
- direct staged copies of `hooks/shared/tool_use_guard_policy.py`

## Validation

Update focused tests to prove:

- source wrappers import `hook_runtime` correctly
- staged wrappers work with only `hooks/hook_runtime/` present
- standalone Claude plugin build contains `hooks/hook_runtime/` and does not
  contain `python_support/`
- plugin `SessionStart` still bootstraps or rebuilds optimize workflow state
- optimize workflow-state tests still cover fresh, resumed, and invalid-state
  behavior through the shared runtime implementation

Run at least these focused checks:

- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin_hooks.py tests/test_optimize_workflow_state.py tests/test_pretooluse_guard_wrappers.py tests/test_agent_hooks.py tests/test_codex_pretooluse_guard.py tests/test_opencode_hook_guard.py`
