# Shared Hook Assets And Trace Summary Design

## Goal

Clarify hook asset ownership now that Codex and Claude share guard policy logic but should keep backend-specific hook wrappers, and reduce trace noise in OpenCode by avoiding duplicated full-command text across `tool_call` and `command` events.

## User-Visible Semantics

- Staged workspace hook file names remain unchanged.
- `--enable-agent-hook` behavior remains unchanged for Codex, Claude, and OpenCode.
- OpenCode trace output should keep emitting both `tool_call` and `command` events, but the `tool_call.summary` field for shell tools should become a short classification summary instead of repeating the full shell command text.

## Design

### Shared Guard Policy Asset

- Keep backend-owned executable wrappers in backend directories:
  - `hooks/codex/` keeps Codex-only templates such as `hooks.json`, `tool_trace_hook.py`, and a Codex `pretooluse_guard.py` wrapper
  - `hooks/claude/` keeps the Claude `pretooluse_guard.py` wrapper
  - `hooks/opencode/` keeps the OpenCode plugin hook
- Move only the backend-agnostic decision logic into `hooks/shared/tool_use_guard_policy.py`.
- Codex and Claude staging code should copy both:
  - the backend-specific `pretooluse_guard.py` wrapper
  - the shared `tool_use_guard_policy.py` module
- The staged workspace filename `pretooluse_guard.py` remains unchanged for both backends, but the file is now a wrapper rather than the full policy implementation.

### OpenCode Trace Summary

- Keep `tool_call` events as the generic tool lifecycle event.
- Keep `command` events as the detailed shell-command event containing the full command text and `command_kind`.
- For OpenCode shell tools, change `tool_call.summary` from the raw command text to a short summary in the form `<tool>: <command_kind>`, for example `bash: benchmark`.
- Non-shell tools keep their existing summary behavior.

## Scope

- No staged workspace filenames change.
- No trace schema fields are removed.
- No analyzer-side schema migration is needed for this iteration.
