# OpenCode Explore Subagent Deny

## Summary

Extend the existing OpenCode workspace config staging so the backend denies both built-in `general` and `explore` task subagents for every `opencode` launch managed by `triton-agent`.

## Goals

- Keep the restriction backend-local inside the existing `.opencode/opencode.json` staging flow.
- Apply the same deny policy to both `general` and `explore`.
- Preserve the current cleanup and existing-config behavior.

## Non-Goals

- Do not change other backends.
- Do not disable additional OpenCode subagents such as `scout`.
- Do not replace the existing warning-and-skip behavior when user-owned `.opencode/opencode.json` already exists.

## Design

Update `src/triton_agent/backends/opencode.py` so `_opencode_workspace_config()` writes:

- `agent.build.permission.task.general = "deny"`
- `agent.build.permission.task.explore = "deny"`
- `agent.plan.permission.task.general = "deny"`
- `agent.plan.permission.task.explore = "deny"`

No command-building behavior changes are needed. The runner should continue to stage the file only when the workspace does not already contain `.opencode/opencode.json`, and should continue removing only the staged file after the run.

## Verification

- Add a failing runner test that asserts the staged config contains both deny entries.
- Re-run `tests.test_opencode_runner` until the updated assertion passes.
