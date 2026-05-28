# Codex JSON Item Trace Design

## Background

`triton-agent optimize --agent codex --log-tools` enables `codex exec --json` and consumes stdout through `CodexJsonOutputFilter`. The current parser recognizes hypothetical `tool_start` and `tool_end` events, but Codex CLI 0.130.0 emits `item.started` and `item.completed` events with nested item types such as `command_execution`, `file_change`, and `agent_message`.

As a result, the trace can prove that native JSON is active while still recording only wrapper-level `agent_invocation` events. The raw worker events remain visible in `optimize.show-output.log`, but they are not normalized into `tool_call`, `command`, `file_access`, or `edit` trace events.

## User-Visible Semantics

When `--log-tools` is passed to a non-interactive Codex run:

- `command_execution` items are recorded as `tool_call` lifecycle events with tool `exec`.
- completed `command_execution` items also produce `command` events.
- read-like shell commands, including PowerShell `Get-Content`, produce best-effort `file_access` events.
- `file_change` items produce `edit` events.
- native JSON items are rendered back into readable show-output text instead of being concatenated as raw JSON.

The trace source remains `codex_native_json`. Durations for `item.started` / `item.completed` pairs are measured by the triton-agent runner receive clock because Codex item events do not carry timestamps.

## Implementation

Extend `src/triton_agent/backends/codex_trace.py` to handle:

- `item.started`
- `item.completed`
- nested `item.type == command_execution`
- nested `item.type == file_change`
- nested `item.type == agent_message`

The parser keeps the existing `tool_start` / `tool_end` support for compatibility with older or future Codex schemas. The `.codex/hooks.json` path remains diagnostic and optional; it is not required for this fix.

## Verification

Add regression tests built from observed Codex CLI 0.130.0 JSONL shapes:

- command start/end writes `tool_call` and `command`
- PowerShell `Get-Content` against staged skill files writes classified `file_access`
- file change start/end writes `edit`
- JSON rendering keeps show-output readable
