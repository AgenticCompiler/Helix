## Summary

Refactor agent hook staging so backend-specific hook preparation moves out of `AgentRunner` and into each backend implementation, matching the existing prepare-context structure used for backend MCP setup.

## User-Visible Semantics

- Codex still stages request-scoped Codex hook files only for the duration of a run when hook guard or tool tracing is enabled.
- OpenCode still stages request-scoped OpenCode hook files only for the duration of a run when hook guard or tool tracing is enabled.
- Claude still does not stage backend-specific hook files.
- Shared execution behavior in `AgentRunner.run()` remains unchanged: launch logging, retries, tracing, output filtering, and process execution continue to work the same way.

## Design

- `AgentRunner.run()` stops preparing agent hooks directly.
- `AgentRunner` exposes helper methods for shared hook options and shared cleanup/logging behavior, but not backend dispatch.
- `CodexRunner` stages and cleans its own hook files inside `_prepare_run_context()`.
- `OpenCodeRunner` stages and cleans its own hook files inside `_prepare_run_context()`.
- `ClaudeRunner` keeps a no-op backend hook path.
- `AgentHookManager` remains the implementation for hook file creation/cleanup, but its entrypoints become backend-specific instead of string-dispatched from base.

## Constraints

- Do not change Codex or OpenCode hook file formats, file locations, or policy contents.
- Do not change the conditions that enable hook guard or tool tracing.
- Do not expand hook support to backends that do not already have it.

## Verification

- Update base runner tests to assert base no longer stages backend hooks itself.
- Add or update Codex and OpenCode runner tests to verify hook files are staged during execution and cleaned afterward.
- Run backend-focused tests plus repository lint, type check, and full test suite.
