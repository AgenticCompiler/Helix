# Run-Eval Shared HTTP MCP Server Design

> Update note: this shared HTTP MCP runtime now applies when `--enable-mcp` is enabled for supported agent-backed run-eval workflows. The legacy script-backed run-eval skill remains the default when the toggle is omitted.

## Summary

`triton-agent` should stop launching a separate stdio MCP process for every code-agent invocation when the staged skills include `triton-npu-run-eval`.

Instead, each top-level `triton-agent` execution should lazily start one shared local HTTP MCP server and point all participating agent invocations at that shared endpoint. This keeps `AgentRequest` simple, centralizes NPU slot management in one place, and avoids per-agent stdio isolation that breaks cross-agent concurrency control.

## Goals

- Keep `AgentRequest.mcp_servers` as a tuple of managed server names only.
- Expose run-eval as MCP tools instead of asking skills to call `run-command.py` directly.
- Share one run-eval MCP server across multiple code-agent invocations within the same `triton-agent` process.
- Keep NPU slot leasing inside the MCP server so agent processes no longer receive `ASCEND_RT_VISIBLE_DEVICES` directly for run-eval-managed flows.
- Support backend-native MCP config emission for `codex`, `claude`, and `opencode`.
- Keep run-eval execution behavior aligned with the existing `run-command.py` subcommands.

## Non-Goals

- No attempt to share one MCP server across multiple independent `triton-agent` processes.
- No generalized transport model in `AgentRequest`; managed servers remain name-based.
- No migration of run-eval business logic into the CLI orchestration layer beyond the server lifecycle and config wiring needed to expose MCP.

## User-Visible Semantics

When a command stages the `triton-npu-run-eval` skill and the selected backend supports request-scoped MCP servers:

- `triton-agent` stages the skill as before.
- `triton-agent` starts a shared local HTTP MCP server on demand.
- The backend receives MCP configuration that points to that local HTTP endpoint.
- The endpoint includes the active workspace as an absolute path in the query string:
  - `http://127.0.0.1:<port>/mcp?workspace=<abs-path>`
- The skill instructs the code agent to use MCP tools instead of shelling out to `run-command.py`.
- The MCP server handles NPU slot leasing for:
  - `run-test-baseline`
  - `run-test-optimize`
  - `run-bench`
  - `profile-bench`

For standalone MCP debugging, `triton-agent` should also provide a dedicated subcommand that starts the shared HTTP run-eval MCP server in the foreground and prints its endpoint information. This command is not agent-backed and does not stage skills; it exists purely so developers can launch the managed MCP runtime directly, inspect its URL, and point external MCP clients at it during debugging.

For unsupported backends, request-scoped MCP usage must fail with a clear validation error instead of silently falling back.

## Why HTTP Instead Of Stdio

With stdio transport, every code-agent invocation launches its own MCP process. In optimize multi-round flows and batch flows, that means NPU concurrency control is split across several independent MCP processes, so the semaphore no longer represents the real shared resource pool.

HTTP solves the immediate problem because one top-level `triton-agent` process can own one shared MCP server instance and all agent invocations in that process can reuse it.

## Workspace Routing

`AgentRequest` should stay transport-agnostic and carry only MCP server names. The active workspace will therefore be conveyed through the backend-emitted MCP URL rather than as extra request fields.

Managed run-eval URLs must take this shape:

- `http://127.0.0.1:<port>/mcp?workspace=<absolute-path>`

The server reads the `workspace` query parameter from the incoming HTTP request and uses it as request context where needed. This keeps the server reusable across multiple workspaces in the same top-level process while preserving the simple request model the user asked for.

Tool arguments should still remain explicit and continue to match the existing `run-command.py` command arguments as closely as possible. File arguments are still expected to be absolute paths after the existing `run-command.py` normalization step.

## Architecture

### 1. Managed MCP Scope

Add a runtime-managed MCP scope in `src/triton_agent/mcp.py`.

Responsibilities:

- Map staged skills to managed MCP server names.
- Lazily start the shared run-eval HTTP server only when a backend actually needs a concrete config entry.
- Reuse one server instance within a process-wide scope.
- Support nested and concurrent usage inside the same process.
- Provide backend-ready resolved server definitions containing HTTP URLs.

The scope must support two usage patterns:

- Implicit per-request scope:
  - if a request needs managed MCP servers and no outer scope exists, create one for that request only
- Explicit top-level scope:
  - batch commands and optimize multi-invocation flows should wrap the whole operation so every nested agent run shares the same server

### 2. Shared HTTP Server

Move the MCP server implementation out of `skills/triton-npu-run-eval/scripts/` and into `src/triton_agent/`.

Responsibilities:

- Build a FastMCP app that exposes the four run-eval tools.
- Serve it over local HTTP using a background ASGI server.
- Parse `workspace` from the HTTP request query string.
- Own the NPU slot pool built from:
  - `TRITON_AGENT_BATCH_NPU_DEVICES`
  - `TRITON_AGENT_BATCH_WORKERS_PER_NPU`
- When those environment variables are absent during standalone MCP debugging, default to device `0` and `1` worker per NPU so the server remains easy to launch manually.
- Lease exactly one NPU slot per tool invocation that needs device-bound execution.
- Invoke the existing run-eval compatibility entrypoint so command semantics stay aligned.

The shared server remains process-local. That is sufficient for the current requirement and avoids premature cross-process coordination design.

### 3. Tool Contract

Expose these tool names:

- `run-test-baseline`
- `run-test-optimize`
- `run-bench`
- `profile-bench`

Each tool keeps the same argument names and behavior shape already documented for the corresponding `run-command.py` subcommand.

Inside the implementation:

- keep command construction close to the current `run-command.py` CLI shape
- preserve remote execution flags
- inject the leased NPU device through environment when invoking the compatibility path

### 4. Backend Config Emission

Supported backends:

- `codex`
- `claude`
- `opencode`

Config shapes:

- Codex:
  - `workdir/.codex/config.toml`
  - `[mcp_servers.<name>]`
  - `url = "http://127.0.0.1:<port>/mcp?workspace=<abs-path>"`
- Claude:
  - `workdir/.claude/mcp.json`
  - passed through `--mcp-config`
  - server entry:
    - `{"type": "http", "url": "..."}`
- OpenCode:
  - `workdir/.opencode/opencode.json`
  - server entry:
    - `{"type": "remote", "url": "..."}`

Other backends should fail fast when `mcp_servers` is requested.

## Lifecycle Placement

The shared scope should wrap the highest layer that meaningfully represents one user command execution.

Required top-level sharing points:

- `run_generation_request(...)`
- `run_convert_request(...)`
- `run_optimize_request(...)`

Additional explicit sharing is needed for batch entrypoints because they invoke several requests in one command:

- `run_gen_eval_batch(...)`
- `run_convert_batch(...)`
- `run_optimize_batch(...)`

This ensures:

- single-item commands still work with a request-local shared scope
- optimize multi-invocation loops reuse one shared server
- batch worker threads reuse one shared server across all workspaces in the same command

## NPU Resource Management

The run-eval MCP server owns the semaphore-equivalent slot pool.

Implementation rules:

- Parse configured NPU devices from `TRITON_AGENT_BATCH_NPU_DEVICES`.
- Parse workers per NPU from `TRITON_AGENT_BATCH_WORKERS_PER_NPU`, defaulting to `1`.
- If `TRITON_AGENT_BATCH_NPU_DEVICES` is unset or blank for standalone server startup, default to a single device pool containing `0`.
- Expand slots as `device * workers_per_npu`.
- Use a blocking thread-safe queue/semaphore style pool to lease and release slots.

Agent-side direct `ASCEND_RT_VISIBLE_DEVICES` injection is no longer the resource-management path for run-eval-managed agent flows.

For batch commands, the environment variables still define overall capacity, but the shared MCP server performs the actual per-tool leasing.

## Compatibility With Existing Skill Scripts

The run-eval MCP server should continue reusing the existing skill-side execution helpers rather than duplicating run-eval logic in the CLI package.

Preferred compatibility approach:

- keep `skills/triton-npu-run-eval/scripts/run-command.py` as the canonical compatibility entrypoint
- invoke it from the shared HTTP MCP server with leased device environment
- continue using `skill_loader` or resource helpers to locate the script from `src/triton_agent`

This preserves the project boundary that workflow logic lives with the skill while the CLI owns orchestration and managed server lifecycle.

## Validation And Errors

Validation requirements:

- if `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is invalid, fail with a clear error naming the variable
- if a backend requests managed MCP servers but no shared MCP scope is active when resolution occurs, the resolver may create a local scope automatically
- if the workspace query parameter is missing or not absolute, the server should fail the tool invocation clearly
- if an unknown managed server name is requested, fail with a `ValueError`

## Testing Strategy

Add or update tests for:

- request model MCP name carriage
- skill-to-managed-server-name mapping
- backend config emission for HTTP URLs
- shared scope reuse across multiple request resolutions
- optimize and batch top-level wrappers sharing a single HTTP server
- run-eval MCP tool registration
- workspace query parsing
- NPU slot reuse across sequential tool calls
- unsupported backend fail-fast behavior

## Migration Notes

- remove the skill-local stdio MCP server as the active implementation path
- keep the skill contract MCP-first
- update tests that currently expect batch-time `ASCEND_RT_VISIBLE_DEVICES` injection into agent requests for MCP-managed flows
