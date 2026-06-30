# Claude Optimize Plugin Hook Design

## Goal

Add a repository script that builds a distributable Claude Code plugin directory for the `optimize --enable-agent-hooks` workflow so users can install the plugin separately from the `triton-agent` CLI and still retain optimize-specific hooks, skills, and agent guidance.

## User-Visible Semantics

- The repository provides a script at `scripts/build-claude-optimize-plugin.py`.
- Running that script produces an already-expanded Claude plugin directory that users can install with Claude's normal plugin flow.
- The generated plugin only supports the Claude optimize workflow that previously depended on `triton-agent optimize --agent claude --enable-agent-hooks`.
- The generated plugin includes:
  - one optimize-focused Claude agent under `agents/`
  - the minimum optimize skill set under `skills/`
  - plugin-scoped Claude hooks under `hooks/`
- The generated plugin does not include standalone `CLAUDE.md` or `prompts.md` files. Their durable guidance content is embedded directly into the optimize agent definition.
- When a Claude session starts with the plugin's optimize agent, the plugin initializes `.triton-agent/` if it is missing.
- When `.triton-agent/state.json` is missing, the plugin attempts a conservative workflow-state bootstrap from durable optimize artifacts already present in the workspace.
- When a Claude session ends, the plugin removes the live `.triton-agent/` runtime directory.
- The plugin never deletes durable optimize artifacts such as `baseline/`, `opt-round-*`, or `triton-agent-logs/`.
- The bundled `ascend-npu-optimize-state` command helpers treat missing workflow state differently by subcommand:
  - `submit-baseline` bootstraps `.triton-agent/state.json` when it is missing, then advances the baseline to accepted state.
  - `start-round` does not bootstrap missing workflow state; it returns a structured error that points the agent at workflow commands such as `submit-baseline`, not at direct edits to internal state files.
  - `set-current-round-state` returns a structured error when workflow state is missing and tells the agent to repair the workflow through state commands before retrying.
  - `submit-round` returns a structured error when workflow state is missing and never silently skips workflow-state completion; the repair guidance stays command-based rather than file-edit based.
  - `start-round` creates the target `opt-round-N/` directory when it does not already exist.
  - `submit-round` returns a structured JSON failure when the target round directory does not exist.

## Problem

The current Claude optimize integration assumes `triton-agent` owns the entire request lifecycle:

- it stages `.claude/skills`
- it stages temporary Claude hook files
- it writes temporary optimize guidance into `CLAUDE.md`
- it creates and later removes `.triton-agent/`
- it bootstraps `.triton-agent/state.json` before hook-guarded optimize work begins

That model breaks down when users copy the generated Claude-facing content into another workspace and launch Claude manually. The copied workspace no longer has the CLI-managed lifecycle that used to:

- prepare `.triton-agent/`
- bootstrap workflow state
- keep optimize guidance visible to Claude
- remove temporary runtime files after the session

For the plugin export use case, those responsibilities must move into the generated Claude plugin itself.

## Scope

This change only covers the Claude optimize hook workflow that corresponds to `optimize --enable-agent-hooks`.

In scope:

- plugin build script
- optimize-only Claude plugin layout
- optimize-only Claude agent packaging
- optimize-only skill packaging
- plugin hook bootstrap and cleanup for `.triton-agent/`
- conservative recovery of missing workflow state
- hook-side diagnostics when workflow state is missing or malformed

Out of scope:

- non-optimize Claude workflows
- Codex, OpenCode, Pi, OpenHands, or TraeCLI plugin exports
- preserving the old temporary `CLAUDE.md` write/restore lifecycle
- automatic restoration of `round_active` state from partially completed rounds
- rebuilding CLI trace/session archive semantics inside the plugin

## Design

### 1. Build Entry Point

Add `scripts/build-claude-optimize-plugin.py`.

The script is the only supported entry point for this packaging flow. It builds one plugin directory tree that is ready for Claude plugin validation and installation.

The script should:

- resolve the optimize skill set from the repository's existing optimize skill staging contract instead of hard-coding a second skill list
- gather the stable optimize guidance text that currently feeds Claude optimize runs
- render one optimize-specific Claude agent that embeds that guidance text
- copy the shared Claude hook guard assets plus new plugin-specific lifecycle hooks
- write a Claude plugin manifest under `.claude-plugin/plugin.json`
- write a short plugin `README.md` describing installation and supported scope

### 2. Generated Plugin Layout

The generated directory should look like this:

```text
<output>/
  .claude-plugin/
    plugin.json
  agents/
    triton-agent-optimize.md
  hooks/
    hooks.json
    pretooluse_guard.py
    tool_use_guard_policy.py
    session_start.py
    session_end.py
    state_bootstrap.py
  skills/
    ...
  README.md
```

The plugin root must not contain standalone `CLAUDE.md` or `prompts.md`.

### 3. Agent Packaging

The generated plugin exports one optimize-focused Claude agent under `agents/triton-agent-optimize.md`.

That file is the durable prompt surface for this plugin. It should inline:

- the current optimize guidance that would otherwise have been written into `CLAUDE.md`
- the stable optimize prompt rules that need to remain visible independently of CLI launch-time prompt assembly

This agent should be the only supported entry point for the plugin's workflow semantics. The plugin hooks should gate themselves to this optimize agent so ordinary Claude sessions are unaffected.

### 4. Hook Ownership

The plugin owns Claude hook configuration through `hooks/hooks.json`.

Use plugin-scoped Claude hook events instead of runner-managed temporary `--settings` files. The hook set should include:

- `SessionStart`
- `SessionEnd`
- `PreToolUse`

`Stop` and `SubagentStop` are not required for the first implementation unless they become necessary for lifecycle correctness. The primary cleanup path should be `SessionEnd`, not `Stop`.

The hook config should invoke plugin-local scripts via `${CLAUDE_PLUGIN_ROOT}` so the generated plugin remains relocatable.

### 5. SessionStart Responsibilities

`hooks/session_start.py` should handle optimize runtime bootstrap for the plugin-managed flow.

Behavior:

- If the current session is not using the plugin's optimize agent, do nothing.
- Ensure the workspace-local `.triton-agent/` directory exists.
- If `.triton-agent/state.json` already exists:
  - validate that it is well-formed enough for optimize-state consumers
  - leave it untouched when valid
  - surface clear additional context when malformed instead of silently replacing it
- If `.triton-agent/state.json` does not exist:
  - attempt conservative recovery from durable workspace artifacts
  - if recovery succeeds, write the recovered workflow state
  - if recovery cannot determine enough information, create only `.triton-agent/` and emit clear guidance about what is still missing

The hook may return `hookSpecificOutput.additionalContext` so Claude sees a short status summary when bootstrap work was necessary.

### 6. Workflow-State Recovery Policy

The plugin must prefer conservative recovery over speculative reconstruction.

#### Recovery inputs

Recovery may inspect:

- `baseline/state.json`
- existing `opt-round-*` directories
- the absence or presence of durable optimize artifacts needed to infer baseline status

Recovery must not depend on transient CLI-only artifacts such as request-scoped trace/session files.

#### Recovery outputs

If `state.json` is missing:

- recover to `awaiting_round_start` only when the workspace clearly contains an accepted baseline
- otherwise recover to `baseline`

The first implementation must not auto-recover to `round_active`.

Rationale:

- an existing `opt-round-N/` directory does not prove that the round was still active when the prior session ended
- treating partially completed or interrupted rounds as active can mis-gate edits into the wrong round directory
- requiring a fresh `start-round` call preserves the current workflow-state contract and keeps recovery deterministic

### 7. PreToolUse Responsibilities

`hooks/pretooluse_guard.py` continues to enforce optimize read/edit protection, but in plugin mode it also needs better missing-state diagnostics.

In addition to current policy checks, the plugin path should distinguish:

- missing `.triton-agent/`
- missing `.triton-agent/state.json`
- malformed `.triton-agent/state.json`
- baseline not yet established
- workflow phase that does not allow the requested edit

This should not degrade into a generic denial when the real problem is missing bootstrap state.

The shared guard policy logic should remain the base decision engine for path protection, while plugin-specific bootstrap diagnosis can live in a small helper layer that runs before or around the current denial logic.

### 8. SessionEnd Responsibilities

`hooks/session_end.py` should remove the live `.triton-agent/` runtime tree for optimize-agent sessions created by this plugin.

Cleanup rules:

- remove `.triton-agent/`
- do not remove `baseline/`
- do not remove `opt-round-*`
- do not remove `triton-agent-logs/`
- do not attempt CLI-style workspace resets

Cleanup should be best-effort and fail-open. A cleanup failure should not make Claude treat the session as failed.

### 9. Skill Selection

The build script should package only the minimum optimize skill set needed by the Claude optimize workflow.

It should derive that set from the repository's existing optimize staging contract so the plugin build stays aligned with normal optimize skill ownership.

Do not introduce a second manually maintained optimize skill list if the existing staging metadata can be reused.

### 10. Guidance Source of Truth

The build script should reuse the repository's existing optimize guidance and prompt-building logic as the source of truth for the embedded agent content whenever possible.

Do not hand-copy or duplicate long guidance blocks into the build script.

The implementation may add a small render helper dedicated to “plugin-safe optimize guidance text” if the current runtime rendering path cannot be reused directly.

## Validation And Testing

Add focused coverage for:

- build script output shape
- generated plugin manifest validity
- generated optimize agent content
- generated optimize skill subset
- `SessionStart` creating `.triton-agent/`
- `SessionStart` conservative recovery to `baseline`
- `SessionStart` conservative recovery to `awaiting_round_start`
- malformed `state.json` detection
- `SessionEnd` removing `.triton-agent/` only
- `PreToolUse` plugin-mode diagnostics for missing or malformed workflow state

Verification should include:

- `claude plugin validate <generated-plugin-dir>`
- repository Python tests covering the new builder and hook helpers

## Risks

### Plugin/runtime divergence

If the plugin build path re-implements optimize guidance or skill selection by hand, it will drift from the CLI-managed optimize path.

Mitigation:

- derive optimize skills from existing staging rules
- reuse existing guidance/prompt rendering helpers where practical

### Over-eager recovery

Recovering to `round_active` without strong evidence can gate edits into stale round directories.

Mitigation:

- first implementation only recovers to `baseline` or `awaiting_round_start`

### Over-broad hook activation

If plugin hooks run for all Claude sessions, the plugin will create `.triton-agent/` in unrelated workspaces.

Mitigation:

- gate lifecycle behavior to the plugin's optimize agent

## Non-Goals

- shipping a general Claude backend export mechanism
- supporting optimize without the plugin's dedicated agent
- mirroring full CLI launch-time prompt composition exactly
- reproducing CLI-managed request archives, traces, or session-id bookkeeping
