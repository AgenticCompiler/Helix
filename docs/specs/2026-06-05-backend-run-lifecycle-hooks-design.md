## Summary

Refactor backend launch lifecycle so `AgentRunner.run()` remains the single shared execution flow, while backend-specific request preparation and cleanup happen through overridable hooks.

## User-Visible Semantics

- Claude, Codex, and OpenCode continue to support request-scoped MCP servers exactly as before.
- Request-scoped MCP config files still exist only for the duration of a single agent run.
- Existing OpenCode workspace config files still win over staged config files and still emit the same warning.
- Backends that do not support request-scoped MCP servers still fail before launching a process.

## Design

- Add a no-op prepare/cleanup context hook to `AgentRunner`.
- `AgentRunner.run()` enters that hook before building the backend command and exits it after the shared execution flow finishes.
- Claude, Codex, and OpenCode move their MCP config staging logic into that hook instead of overriding `run()`.
- OpenHands keeps its custom `run()` because it owns different process semantics, not just request preparation.

## Constraints

- Do not change backend command construction semantics.
- Do not change MCP config file formats or locations.
- Do not change hook-manager behavior, retry behavior, tracing, or output filtering in `AgentRunner.run()`.

## Verification

- Add a base-runner lifecycle test proving backend prepare/cleanup wraps command building and process execution.
- Re-run backend tests that cover staged MCP config creation, warning behavior, and cleanup for Claude, Codex, and OpenCode.
