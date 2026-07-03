## Summary

Share the Python `PreToolUse` wrapper logic used by Claude, Codex, and the standalone Claude optimize plugin, while keeping separate entry scripts for their different host environments.

## Problem

The repository currently keeps multiple Python `pretooluse_guard.py` wrappers with overlapping logic for:

- loading stdin JSON
- fail-open error handling
- loading `tool_use_guard_policy.py`
- emitting Claude/Codex denial JSON

That duplication makes behavior drift likely, especially when the standalone Claude plugin is built into a self-contained artifact.

## Design

Add one shared Python helper module under `hooks/shared/` that owns the common wrapper mechanics:

- loading nearby shared modules
- loading `policy.json` when a wrapper uses file-backed policy
- calling `deny_reason_for_tool_use`
- emitting standardized denial output
- preserving fail-open behavior

Keep separate entry wrappers for:

- `hooks/claude/pretooluse_guard.py`
- `hooks/codex/pretooluse_guard.py`
- `hooks/claude_plugin/pretooluse_guard.py`

Those entry scripts should only keep host-specific behavior, such as:

- whether policy comes from `--policy` or is computed from `cwd`
- plugin-only `agent_type` gating
- plugin-only workflow-state prechecks
- host-specific fail-open log prefixes

## Distribution

Any staging or build flow that currently copies `tool_use_guard_policy.py` into a runtime hook directory must also copy the new shared helper module so standalone artifacts do not depend on the source tree.

## Scope

This change does not alter guard-policy semantics. It only reduces wrapper duplication and keeps staged/plugin artifacts in sync with the shared wrapper runtime.
