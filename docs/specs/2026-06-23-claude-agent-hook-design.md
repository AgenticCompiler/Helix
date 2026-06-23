# Claude Agent Hook Design

## Goal

Add `--enable-agent-hook` support to the Claude backend with the same request-scoped semantics already used by Codex and OpenCode.

## User-Visible Semantics

- `triton-agent optimize --agent claude --enable-agent-hook` should stage temporary Claude hook files in the target workspace before launching Claude.
- The staged hook should be removed after the Claude run exits.
- Without `--enable-agent-hook`, the Claude backend should behave exactly as it does today.
- This change must not overwrite or merge user-owned `.claude/settings.json` or `.claude/settings.local.json`.

## Design

- Stage a temporary request-scoped Claude settings file under `.claude/triton-agent-hooks/settings.json`.
- Pass that file to Claude with `--settings <path>`.
- Use a Claude-specific `pretooluse_guard.py` wrapper while keeping the guard policy logic shared with other backends.
- Make the shared guard policy logic policy-driven for protected skill roots so it can protect `.claude/skills/*/scripts/` in addition to the existing Codex and OpenCode staged skill roots.
- Stage these Claude-owned files:
  - `.claude/triton-agent-hooks/settings.json`
  - `.claude/triton-agent-hooks/policy.json`
  - `.claude/triton-agent-hooks/pretooluse_guard.py`
  - `.claude/triton-agent-hooks/tool_use_guard_policy.py`

## Scope

- This first Claude implementation only adds native hook enforcement for `--enable-agent-hook`.
- Tool tracing for Claude should continue using the existing `stream-json` output filter path rather than hook-based tracing.
- No plugin packaging is needed for Claude in this change.
