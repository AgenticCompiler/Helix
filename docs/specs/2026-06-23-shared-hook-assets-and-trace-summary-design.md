# Shared Hook Assets And Trace Summary Design

## Goal

Clarify hook asset ownership now that the Python `PreToolUse` guard is shared by Codex and Claude, and reduce trace noise in OpenCode by avoiding duplicated full-command text across `tool_call` and `command` events.

## User-Visible Semantics

- Staged workspace hook file names remain unchanged.
- `--enable-agent-hook` behavior remains unchanged for Codex, Claude, and OpenCode.
- OpenCode trace output should keep emitting both `tool_call` and `command` events, but the `tool_call.summary` field for shell tools should become a short classification summary instead of repeating the full shell command text.

## Design

### Shared Guard Asset

- Move the shared Python guard template from `hooks/codex/pretooluse_guard.py` to `hooks/shared/pretooluse_guard.py`.
- Keep backend-owned assets in backend directories:
  - `hooks/codex/` keeps Codex-only templates such as `hooks.json` and `tool_trace_hook.py`
  - `hooks/opencode/` keeps the OpenCode plugin hook
- Codex and Claude staging code should both copy the guard from `hooks/shared/pretooluse_guard.py` into their backend-local staged hook directories.

### OpenCode Trace Summary

- Keep `tool_call` events as the generic tool lifecycle event.
- Keep `command` events as the detailed shell-command event containing the full command text and `command_kind`.
- For OpenCode shell tools, change `tool_call.summary` from the raw command text to a short summary in the form `<tool>: <command_kind>`, for example `bash: benchmark`.
- Non-shell tools keep their existing summary behavior.

## Scope

- No staged workspace filenames change.
- No trace schema fields are removed.
- No analyzer-side schema migration is needed for this iteration.
