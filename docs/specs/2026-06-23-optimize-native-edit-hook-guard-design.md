# Optimize Built-In Edit Tool Hook Guard Design

## Goal

Implement the previously deferred workflow-state-aware built-in edit tool guard for optimize runs that use `--enable-agent-hook`.

This design covers only backend built-in edit tools. It does not add Bash write interception in this first version.

## User-Visible Semantics

- The built-in edit tool guard is enabled only for optimize runs launched with `--enable-agent-hook`.
- Without `--enable-agent-hook`, optimize keeps today's behavior.
- The guard reads `.helix/state.json` from the operator workspace root and uses it as the phase authority.
- If `.helix/state.json` is missing or invalid, built-in edit tools fail with an agent-facing recovery hint that asks the runner to restart the optimize session.
- During `baseline`, built-in edit tools may modify only the baseline-minimal file set:
  - the source operator recorded in workflow state
  - root-level `test_*.py`, `differential_test_*.py`, and `bench_*.py`
  - files under `baseline/`
- During `awaiting_round_start`, built-in edit tools may not modify any files. The denial message must tell the agent to use `triton-npu-optimize-start-round` before editing.
- During `round_active`, built-in edit tools may modify only files under the active `opt-round-N/` directory.
- When a `round_active` edit is denied for targeting files outside the active round directory, the denial message must tell the agent to use `triton-npu-optimize-submit-round` when the current round is ready to be submitted.
- Agent-facing denial messages should stay operational and recovery-oriented. Version-scope caveats such as “Bash file writes are not blocked in this first version” belong in design docs, not in denial text shown to the agent.

## Backend Scope

- Codex:
  - extend the shared policy module plus the Codex `pretooluse_guard.py` wrapper to recognize `Write`, `Edit`, and `MultiEdit`
  - stage the Codex wrapper on `PreToolUse` for the same matcher set, so edit tools are blocked before execution rather than traced after the fact
- OpenCode:
  - extend `hooks/opencode/helix-hook-guard.js` built-in edit enforcement in `tool.execute.before`
  - recognize the existing built-in edit tool aliases already traced there (`write`, `edit`, `patch`, `update`, `multiedit`, `multi_edit`)

## Path Rules

- Resolve candidate edit paths relative to the tool cwd, then resolve symlinks before policy checks.
- Treat the operator workspace root as the only workflow-state discovery root. No ancestor search is allowed.
- Keep `.helix/`, staged skill implementation files, and `helix-logs/` outside the built-in-edit allowlist in every phase by omission from the phase allowlists.

## Testing

- Add Codex guard tests for:
  - `baseline` allow and deny cases
  - `awaiting_round_start` denial with a start-round hint
  - `round_active` allow and deny cases
  - missing or invalid workflow state denial
  - staged Codex hooks invoking `pretooluse_guard.py` on `PreToolUse`
- Add matching OpenCode guard tests for the same phase behaviors.
