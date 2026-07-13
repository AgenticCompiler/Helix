# Claude Plugin Subagent Hook Lifecycle Design

## Summary

Keep Claude optimize-plugin runtime state ownership on both session-scoped and subagent-scoped hooks.

Claude Code may run the optimize plugin agent directly as the active agent or may invoke it from the plugin `agents/` directory as a subagent. Therefore `.helix/state.json` should be bootstrapped from either `SessionStart` or `SubagentStart`, and cleaned up from either `SessionEnd` or `SubagentStop`. Those lifecycle actions must be idempotent: when valid runtime state already exists, later bootstrap calls should reuse it instead of overwriting it, and later cleanup calls should safely no-op when the runtime tree is already gone. Missing workflow state should no longer block ordinary workspace edits by itself. When workflow state is absent, edit guards should skip workflow-phase gating while still protecting internal runtime paths such as `.helix/`, staged hook files, staged skill scripts, and `helix-logs/`.

## Goals

- Make optimize runtime bootstrap line up with the actual Claude plugin agent lifecycle.
- Support both direct optimize-agent startup and optimize-agent subagent startup.
- Rename the optimize plugin agent identifier and file from `helix-optimize` to `helix-optimizer`.
- Stop blocking baseline repair edits only because `.helix/state.json` is missing.
- Keep internal runtime paths protected even when workflow state is absent.
- Preserve existing baseline / round workflow-state gating once state exists.

## Non-Goals

- Do not redesign optimize phase names, round semantics, or `ascend-npu-optimize-state` command behavior.
- Do not infer round state from durable artifacts inside edit guards.
- Do not remove the shared tool-use guard layer.
- Do not weaken protection for `.helix/`, staged hook implementation files, staged skill `scripts/`, or `helix-logs/`.

## User-Visible Behavior

### Optimize worker startup

- When Claude starts the optimize plugin agent directly, `SessionStart` should ensure the workspace-local `.helix/` directory exists and bootstrap `.helix/state.json`.
- When the plugin optimize agent is spawned as a Claude subagent, `SubagentStart` should do the same.
- For a fresh optimize workspace, bootstrap a `baseline` phase state.
- For a resumable optimize workspace with a reusable canonical baseline, bootstrap the existing resumable state exactly as current shared bootstrap logic already defines.
- If valid runtime state already exists, bootstrap should reuse it and skip rewriting it.

### Optimize worker shutdown

- When a directly started optimize-agent session ends, `SessionEnd` should remove the live `.helix/` runtime tree.
- When an optimize subagent finishes, `SubagentStop` should remove the same runtime tree when it owns it.
- Cleanup should remain best-effort and must not delete durable optimize artifacts such as `baseline/`, `opt-round-*`, or `opt-note.md`.
- Cleanup should be idempotent and must tolerate the runtime tree already being absent.

### Missing workflow state during editing

- If `.helix/state.json` is missing, malformed, or otherwise unavailable, plugin edit guards should not deny ordinary edits to workspace files only because workflow state is missing.
- The edit operation should continue through the normal shared path checks.
- Internal runtime paths must still remain denied.

### Workflow gating when state exists

- Once `.helix/state.json` exists and is valid, the existing phase-based behavior remains in force:
  - `baseline` permits baseline repair edits.
  - `awaiting_round_start` denies round edits until `start-round`.
  - `round_active` restricts edits to the active `opt-round-N/` and allowed top-level narrative files.

## Hook Responsibility Split

### `SubagentStart`

Owns optimize runtime bootstrap for optimize-agent subagent launches.

- Resolve the workspace from the hook payload.
- Ignore unrelated subagents.
- Reuse the existing shared bootstrap helper so resumable/fresh state semantics stay centralized.
- Surface short diagnostic context only when bootstrap needs to warn about invalid resumable residue.
- If runtime state already exists and is valid, reuse it.

### `SubagentStop`

Owns optimize runtime cleanup for the plugin optimize agent.

- Resolve the workspace from the hook payload.
- Ignore unrelated subagents.
- Remove `.helix/` only.

### `SessionStart` and `SessionEnd`

Retain `.helix/` lifecycle ownership for direct optimize-agent sessions.

- `SessionStart` should continue to bootstrap optimize runtime state when the optimize plugin agent is started directly.
- `SessionEnd` should continue to clean up runtime state for that direct-session path.
- Session bootstrap and cleanup must be idempotent so they can coexist safely with subagent lifecycle hooks.

### `PreToolUse`

Remains the enforcement layer, but missing-state handling changes.

- Keep denying protected internal runtime paths before any workflow-state decision.
- If workflow state is missing, skip workflow-phase gating instead of denying ordinary edits.
- If workflow state exists, continue using the current shared phase-based policy.

## Implementation Shape

- Rename the generated optimize agent file and identifier from `agents/helix-optimize.md` / `helix-optimize` to `agents/helix-optimizer.md` / `helix-optimizer`.
- Update `hooks/claude_plugin/hooks.json` to register `SessionStart`, `SessionEnd`, `SubagentStart`, and `SubagentStop`.
- Add thin wrappers:
  - `hooks/claude_plugin/subagent_start.py`
  - `hooks/claude_plugin/subagent_stop.py`
- Extend `hooks/claude_plugin/state_bootstrap.py` with helper logic that can identify the optimize plugin subagent from subagent hook payloads.
- Change `hooks/claude_plugin/pretooluse_guard.py` so missing workflow state is not an immediate deny reason.
- Change `src/hook_runtime/tool_use_decision.py` so built-in edit checks treat missing workflow state as "skip workflow gating" rather than "deny", while preserving protected-path denial.

## Failure Handling

- If subagent bootstrap throws unexpectedly, the hook should still fail open as current wrappers do, but `PreToolUse` should no longer compound that failure by denying ordinary edits solely because state is absent.
- If workflow state is malformed, protected internal paths remain denied, but ordinary workspace edits may proceed until the worker explicitly re-establishes state.
- Workflow-state repair commands such as `submit-baseline` remain the authority for bringing the state machine back to a valid phase.

## Testing Strategy

- Add Claude plugin hook tests proving `SubagentStart` creates fresh baseline state for the optimize plugin subagent.
- Add Claude plugin hook tests proving `SubagentStop` removes `.helix/` and leaves durable artifacts alone.
- Replace current plugin tests that assume `SessionStart` owns optimize bootstrap.
- Add guard tests proving:
  - missing workflow state no longer blocks normal workspace edits
  - protected runtime paths are still denied when workflow state is missing
  - existing `baseline`, `awaiting_round_start`, and `round_active` gating still works unchanged once state exists

## Rationale

The current design assumes session-level startup is always the authoritative moment to create optimize workflow state. That is too brittle for plugin-defined optimize workers that Claude may invoke as subagents inside an already-running parent session. At the same time, direct optimize-agent startup still needs the session-level path. Keeping both lifecycle entrypoints and making bootstrap/cleanup idempotent matches both execution modes, while changing missing-state edit behavior prevents bootstrap gaps from blocking baseline repair work before the agent has a chance to recover.
