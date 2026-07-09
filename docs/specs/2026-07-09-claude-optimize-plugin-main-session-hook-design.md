# Claude Optimize Plugin Main-Session Hook Design

## Summary

Treat the standalone Claude optimize plugin as optimize-only at the plugin level. Main-session hooks should no longer try to infer whether the current Claude session is an optimize session from `agent_type` or similar payload fields. Instead, `SessionStart`, `SessionEnd`, and `PreToolUse` should always apply the optimize workflow lifecycle and guard behavior whenever this plugin is installed and its hooks run. Subagent lifecycle hooks should keep their current precise `subagent_type` and `agent_id` checks because they still need ownership-aware cleanup.

## Problem

The current Claude plugin hook flow assumes the optimize plugin can identify optimize sessions from hook payload metadata such as `agent_type`. That assumption is not stable for direct main-session usage: when the plugin is used in the active Claude session instead of a subagent, `agent_type` may be empty, so:

- `SessionStart` skips `.triton-agent/` bootstrap
- `PreToolUse` skips optimize workflow guard enforcement
- `SessionEnd` skips runtime cleanup

This leaves direct optimize sessions without the workflow state and protections that the plugin is supposed to provide.

## Goals

- Make direct Claude main-session optimize usage reliable even when hook payloads omit `agent_type`.
- Keep the standalone Claude plugin behavior aligned with its real deployment model: this plugin is used only for our optimize workflow.
- Preserve precise owner-aware cleanup for optimize subagents.
- Minimize implementation risk by changing only Claude plugin hook gating, not workflow state semantics.

## Non-Goals

- Do not redesign optimize workflow phases or `.triton-agent/state.json` contents.
- Do not change subagent ownership recording or owner matching rules.
- Do not make this plugin support mixed optimize and non-optimize Claude agents in the same installation.
- Do not change shared hook-runtime path protection semantics beyond routing more direct sessions through the existing guard.

## User-Visible Behavior

### Direct Claude main-session usage

When this plugin is installed and Claude starts a session in a workspace:

- `SessionStart` always bootstraps optimize runtime state for the workspace.
- `PreToolUse` always enforces the optimize path and workflow guard for the workspace.
- `SessionEnd` always removes the live `.triton-agent/` runtime tree.

This behavior no longer depends on `agent_type`, `subagent_type`, or other optimize-session inference.

### Claude subagent usage

Subagent lifecycle hooks remain selective:

- `SubagentStart` still runs only for the optimize subagent type.
- `SubagentStop` still cleans up only when the recorded owner matches both `agent_id` and `agent_type`.

This keeps precise cleanup behavior for subagent-owned runtime trees and avoids broadening ownership side effects.

## Design

### Optimize-only plugin contract

Treat the built Claude plugin as an optimize-only plugin contract rather than a general Claude plugin that happens to contain optimize helpers. Under that contract, direct session hooks should assume that any session using this plugin is an optimize workflow session.

### Hook gating changes

- Remove `should_manage_payload(...)` gating from:
  - `hooks/claude_plugin/session_start.py`
  - `hooks/claude_plugin/session_end.py`
  - `hooks/claude_plugin/pretooluse_guard.py`
- Keep an explicit optimize-subagent helper only for `hooks/claude_plugin/subagent_start.py`, where the hook still needs explicit optimize-subagent type matching before recording runtime ownership.

### Helper responsibility cleanup

`should_manage_payload(...)` currently reads like a universal plugin gate, but after this change it is no longer appropriate for main-session hooks. The implementation should replace it with a narrower helper such as `is_optimize_subagent_payload(...)` and remove it from direct-session hook paths so the optimize-only plugin contract is obvious in code.

## Testing Strategy

Add regression coverage proving that direct main-session behavior no longer depends on `agent_type`:

- `SessionStart` bootstraps `.triton-agent/state.json` when the payload only contains `cwd`.
- `PreToolUse` still denies protected runtime reads and still applies workflow gating when the payload only contains `cwd`, `tool_name`, and `tool_input`.
- `SessionEnd` removes `.triton-agent/` when the payload only contains `cwd`.

Keep existing subagent tests to prove:

- unrelated subagents are still ignored
- owner-aware `SubagentStop` cleanup still requires matching `agent_id` and `agent_type`

## Rationale

This plugin is not being used as a general-purpose Claude plugin. It is installed specifically to provide optimize workflow behavior. Under that deployment model, trying to infer optimize intent from optional hook payload metadata adds brittleness without providing useful protection. Making main-session hooks unconditional is the simplest behavior that matches real usage, while keeping subagent hooks selective preserves the only place where payload identity still matters.
