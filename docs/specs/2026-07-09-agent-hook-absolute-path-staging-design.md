# Agent Hook Absolute Path Staging Design

## Goal

Make runner-staged Claude and Codex hook configs use absolute paths for staged
hook scripts and staged `policy.json` files so hook execution does not depend on
the agent process working directory.

## User-Visible Semantics

- When `--enable-agent-hooks` stages Claude hooks, the generated
  `.claude/triton-agent-hooks/settings.json` invokes the staged guard script with
  absolute paths.
- When `--enable-agent-hooks` or tool tracing stages Codex hooks, the generated
  `.codex/hooks.json` invokes the staged trace and guard scripts with absolute
  paths.
- OpenCode hook behavior stays unchanged.
- Claude plugin hook behavior stays unchanged.

## Problem

The current Claude and Codex staging flows copy hook config templates into the
workspace without rewriting the relative hook paths embedded in those templates.
That leaves staged configs pointing at `.claude/...` and `.codex/...` relative
paths. If the host agent runs those hook commands from a different working
directory, hook execution can fail even though the staged files exist.

## Scope

In scope:

- rewriting staged Claude hook config paths to absolute workspace paths
- rewriting staged Codex hook config paths to absolute workspace paths
- focused regression tests for the staged config payloads

Out of scope:

- changing OpenCode hook policy loading
- changing Claude plugin hook commands
- changing staged hook script contents
- changing hook guard policy semantics

## Design

Keep the repository hook config templates as structure templates, but stop
copying the Claude and Codex config files verbatim during staging.

For Claude:

- keep `hooks/claude/settings.json` as a checked-in structure template
- express staged hook paths in that template as
  `${CLAUDE_PROJECT_DIR}/.claude/triton-agent-hooks/...`
- during staging, replace `${CLAUDE_PROJECT_DIR}` with the resolved workspace
  path and write the rendered JSON to `.claude/triton-agent-hooks/settings.json`

For Codex:

- keep `hooks/codex/hooks.json` as a checked-in structure template
- express staged hook command paths in that template as
  `"${CODEX_PROJECT_DIR}/.codex/triton-agent-hooks/..."`
- during staging, replace `${CODEX_PROJECT_DIR}` with the resolved workspace
  path and write the rendered JSON to `.codex/hooks.json`

This keeps the source templates readable, makes the placeholder ownership
explicit in version control, preserves shell-safe quoted Codex command paths,
and still makes the prepared workspace artifact independent of runtime cwd.

## Testing

- Update Claude hook staging tests to assert the staged `settings.json` contains
  absolute paths instead of matching the source template byte-for-byte.
- Update Codex hook staging tests to assert the staged `hooks.json` command
  strings contain absolute paths for both hook scripts and `policy.json`.
