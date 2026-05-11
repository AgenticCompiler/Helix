# OpenCode Plugin Hook Guard Design

## Goal

Extend the opt-in agent hook guard to the OpenCode backend by staging a temporary project-local OpenCode plugin that blocks redundant reads outside the target workspace and reads of staged skill implementation scripts.

The behavior should match the Codex hook guard at the user contract level: hooks are disabled by default, enabled only by `optimize --enable-agent-hooks`, staged into the target workspace before the agent starts, and removed after the agent exits.

## User-Visible Behavior

- `optimize --enable-agent-hooks --agent opencode` stages an OpenCode plugin into the target workspace before launching OpenCode.
- Without `--enable-agent-hooks`, OpenCode launches exactly as it does today.
- The guard blocks shell read commands that target paths outside the target workspace.
- The guard blocks shell read commands that target staged skill implementation scripts under `.opencode/skills/*/scripts/**`.
- The denial message tells the agent to stay within the workspace and use skill instructions or documented command interfaces instead of reading implementation scripts.
- Cleanup removes only paths created by the current run.
- Existing user-owned OpenCode plugin or hook paths are never overwritten.

## OpenCode Plugin Contract

OpenCode loads project-level local plugins from `.opencode/plugins/` at startup. A plugin exports an async function and can subscribe to `tool.execute.before` to inspect and modify tool calls before execution. OpenCode's plugin documentation shows this hook can throw an error to prevent a tool call, including for file-read protection examples.

This means the OpenCode hook guard should be implemented as a staged project plugin rather than as a CLI hook config file.

One backend-specific constraint is important: the current OpenCode runner always passes `--pure`, and local `opencode --help` describes `--pure` as running without external plugins. Therefore, when `request.enable_agent_hooks` is true for OpenCode, the runner must omit `--pure` so the project plugin can load. Default OpenCode runs should keep `--pure` to preserve the existing isolated behavior.

## Staged Files

Repository templates should live under:

```text
hooks/opencode/
```

The staged workspace should receive:

```text
.opencode/plugins/triton-agent-hook-guard.js
.opencode/triton-agent-hooks/policy.json
```

The plugin source is static and copied from the template directory. The policy is rendered for the current workspace so absolute path checks are deterministic.

The policy shape should mirror the Codex policy while using OpenCode paths:

```json
{
  "workspace_root": "/absolute/path/to/workspace",
  "allow_read_roots": ["/absolute/path/to/workspace"],
  "deny_read_globs": [
    "/absolute/path/to/workspace/.opencode/skills/*/scripts/**"
  ],
  "deny_message": "This read is blocked by triton-agent workspace policy. Stay within the current workspace and do not inspect staged skill implementation files under .opencode/skills/*/scripts/. Use the skill instructions and documented command interface instead."
}
```

## Manager Lifecycle

`AgentHookManager.prepare_hooks()` should gain an `opencode` branch alongside the existing `codex` branch.

The OpenCode branch should:

- Refuse to overwrite `.opencode/plugins/triton-agent-hook-guard.js`.
- Refuse to overwrite `.opencode/triton-agent-hooks/`.
- Create parent directories as needed.
- Copy the plugin template.
- Render `policy.json`.
- Track created paths in cleanup order.
- Clean up tracked paths after the agent exits.

The manager should not attempt to merge with user-owned OpenCode plugins. A later change can introduce merge semantics if OpenCode exposes a safer project-plugin composition mechanism for this use case.

## Guard Behavior

The plugin should subscribe to `tool.execute.before` and evaluate shell-like tool calls. The current expected OpenCode shell tool name is `bash`, matching the plugin documentation examples.

The guard should:

- Ignore non-shell tool calls.
- Inspect `output.args.command` when it is a string.
- Detect common read-oriented commands: `cat`, `sed`, `head`, `tail`, `less`, `more`, `awk`, `rg`, `python`, and `python3`.
- Extract path-like tokens from shell input and common Python one-liners.
- Resolve relative paths against the tool call cwd when available, otherwise the policy workspace root.
- Deny resolved paths outside `allow_read_roots`.
- Deny resolved paths matching `deny_read_globs`.
- Throw `new Error(policy.deny_message)` to block the tool call.

The guard is intentionally conservative. It should block clear read attempts without trying to parse every shell construct.

## Runner Behavior

OpenCode command construction should keep the existing command shape for normal runs:

```text
opencode run --dir <workspace> --pure --thinking <prompt>
opencode <workspace> --pure --thinking --prompt <prompt>
```

When hooks are enabled, the same commands should omit `--pure`:

```text
opencode run --dir <workspace> --thinking <prompt>
opencode <workspace> --thinking --prompt <prompt>
```

No new CLI flag is needed. `--enable-agent-hooks` already maps to `AgentRequest.enable_agent_hooks` for optimize runs.

## Testing

- Add OpenCode staging tests to `tests/test_agent_hooks.py`.
- Add OpenCode plugin guard tests that import the JavaScript plugin through Node and exercise allowed reads, outside-workspace reads, protected `.opencode/skills/*/scripts/**` reads, and non-shell tool calls.
- Add OpenCode runner tests showing `--pure` remains in default commands and is omitted when hooks are enabled.
- Keep existing Codex hook guard tests unchanged.
- Run focused backend and hook tests, then the standard repository verification commands documented in `README.md`.

## Scope Boundaries

- Do not change OpenCode skill staging under `.opencode/skills`.
- Do not add hooks to generation, convert, or batch commands in this change.
- Do not modify user-level OpenCode config or global plugin directories.
- Do not treat this as a security sandbox. This is an agent-behavior guardrail.
- Do not move workflow behavior from skills into CLI code.
