# OpenCode General Subagent And Continuous Round Serialization

## Summary

- Tighten the `continuous` optimize prompt so one session completes rounds strictly in order.
- Explicitly forbid using subagents to execute multiple optimize rounds in parallel.
- Stage an OpenCode workspace config at `.opencode/opencode.json` that denies the built-in `general` subagent through `permission.task`.

## Goals

- Reduce prompt ambiguity in `continuous` optimize mode about whether multiple rounds may be advanced at once.
- Prevent OpenCode from automatically dispatching the built-in `general` subagent for parallel multi-round execution.
- Keep the change backend-local, additive, and safe to clean up after the current run.

## Non-Goals

- Do not disable all OpenCode subagents.
- Do not change `checked` or `supervised` round ownership semantics.
- Do not overwrite or merge with a user-owned `.opencode/opencode.json`.

## Design

### Continuous Prompt

Add explicit wording to the `continuous` optimize prompt, and its resume prompt path, that:

- only one optimize round may be active at a time
- rounds must be completed sequentially
- subagents may help with supporting analysis only
- subagents must not be used to implement or advance multiple rounds in parallel

This keeps the session model aligned with the existing round gate language that already requires the current round to pass `check-round` before the next round begins.

### OpenCode Workspace Config

When the OpenCode backend launches, stage a workspace-local config file at:

- `.opencode/opencode.json`

Write a minimal config that targets the built-in primary agents we rely on and denies the built-in `general` subagent via `permission.task.general = "deny"`.

The staged file should:

- include the OpenCode JSON schema URL
- deny only `general`
- leave `explore` and `scout` untouched

If `.opencode/opencode.json` already exists, fail explicitly instead of overwriting or merging it.

### Cleanup

Remove only the staged `opencode.json` file created by the current run. Do not delete or replace any pre-existing user-owned OpenCode config.
