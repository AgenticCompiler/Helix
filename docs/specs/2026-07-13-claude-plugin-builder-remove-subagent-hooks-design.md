# Claude Plugin Builder Remove Subagent Hooks Design

## Summary

Remove Claude plugin subagent lifecycle packaging from the Helix repository builder. The built plugin should keep only the main-session optimize lifecycle hooks and should stop advertising subagent usage.

## Problem

The current builder still packages `SubagentStart` and `SubagentStop` hook support and the generated plugin README still describes "Use as a subagent". If the plugin should no longer support that path, keeping those assets creates dead code, extra tests, and misleading user-facing instructions.

## Goals

- The built Claude plugin no longer ships `subagent_start.py` or `subagent_stop.py`.
- The built hook manifest no longer registers `SubagentStart` or `SubagentStop`.
- The generated plugin README no longer describes subagent usage.
- Remove builder-local code and tests that only exist for the packaged subagent lifecycle.

## Non-Goals

- Do not change `SessionStart`, `SessionEnd`, or `PreToolUse` behavior for direct optimize sessions.
- Do not change optimize workflow state semantics outside the removed subagent path.
- Do not refactor unrelated Claude plugin packaging code.

## Design

- Delete `hooks/claude_plugin/subagent_start.py` and `hooks/claude_plugin/subagent_stop.py`.
- Remove `SubagentStart` and `SubagentStop` entries from `hooks/claude_plugin/hooks.json`.
- Remove `plugin-owner` and subagent-owner helper logic from `hooks/claude_plugin/state_bootstrap.py`.
- Remove unused builder arguments that only implied subagent support.
- Update the generated plugin README to describe direct optimize-agent startup only.
- Update hook and packaging tests to assert the subagent assets are gone.

## Testing Strategy

- Update packaging tests so the built plugin must omit the subagent hook files and manifest entries.
- Remove hook tests that only exercised the deleted subagent lifecycle scripts.
- Re-run focused Claude plugin test modules to confirm direct-session behavior still passes without subagent packaging.
