# Agent Hook Compiler Source Allow Root Design

## Goal

When optimize runs enable compiler source analysis and agent hooks at the same time,
the staged hook policy should allow reads from the resolved compiler source checkout
for that run.

## Problem

`build_optimize_request()` resolves the compiler source checkout and stores it on
`AgentRequest.compiler_source_path`, but `AgentHookManager` currently renders policy
files with only the workspace root in `allow_read_roots`. That makes the hook treat
the compiler source checkout as an outside-workspace read even though the optimize
prompt explicitly authorizes compiler-source analysis.

## Design

- Keep the allowlist narrow: add only the resolved `request.compiler_source_path`
  for the current run, not the entire `~/.triton-agent/compiler-sources/` parent.
- Extend `AgentHookManager.prepare_hooks()` with an optional list of extra allowed
  read roots and thread those roots into both Codex and OpenCode policy rendering.
- Update the shared runner hook staging call to pass the current request's compiler
  source checkout when present.
- Leave deny globs unchanged so `triton-agent-logs/` and staged skill scripts stay
  protected even if extra allow roots are present.

## Verification

- Policy generation tests should show the compiler source checkout appended to
  `allow_read_roots`.
- Runner tests should show `request.compiler_source_path` being passed into hook
  staging.
- Guard tests should allow a shell read from a compiler source file when the policy
  includes that checkout as an extra allowed root.
