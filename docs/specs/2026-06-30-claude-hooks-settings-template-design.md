## Summary

Switch staged Claude hook configuration from runtime-generated `settings.json` to a fixed template file copied into the workspace.

## Problem

`prepare_claude_hooks()` currently generates `settings.json` in Python even though the staged hook wiring is structurally stable.

That makes the Claude hook staging path less symmetric with Codex, which already uses a checked-in `hooks.json` template.

## Design

Add a checked-in template file at `hooks/claude/settings.json` that uses stable workspace-relative hook paths:

- `.claude/helix-hooks/pretooluse_guard.py`
- `.claude/helix-hooks/policy.json`

Update `prepare_claude_hooks()` to copy that template into the staged hook directory instead of synthesizing JSON in code.

## Constraints

- Keep the staged settings path unchanged: `.claude/helix-hooks/settings.json`
- Keep the staged hook command contract unchanged apart from replacing absolute paths with stable relative paths
- Do not change guard or policy semantics

## Scope

This change is limited to Claude staged hook configuration. It does not affect Codex, OpenCode, or standalone Claude plugin hook packaging.
