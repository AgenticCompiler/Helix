# Run-Eval MCP Server Design

## Summary

- Add a new FastMCP stdio server under `skills/triton-npu-run-eval/scripts/` for the run-eval workflow.
- Expose four MCP tools whose names and arguments stay aligned with the existing run-eval command surface:
  - `run-test-baseline`
  - `run-test-optimize`
  - `run-bench`
  - `profile-bench`
- Move NPU device allocation and concurrency control into that MCP server so agent prompts and skills no longer manage `ASCEND_RT_VISIBLE_DEVICES` directly.
- Extend agent-backed requests with a generic `mcp_servers` field and have supported backends materialize backend-specific MCP config files inside their own workspace directories.
- Keep `skills/triton-npu-run-eval/scripts/run-command.py` as a compatibility and manual CLI entrypoint, but stop making it the primary agent-facing contract.

## Goals

- Make run-eval execution available to supported code-agent backends through MCP tools instead of direct script calls in skills.
- Keep the run-eval skill as the source of truth for run-eval behavior; do not move run-eval workflow logic into `src/triton_agent`.
- Centralize NPU resource management in one place:
  - parse the assigned device pool once at server startup
  - lease one concrete device per tool invocation
  - release that device when the invocation completes
- Reuse the existing skill-local run-eval Python implementation as much as practical instead of rewriting test, benchmark, and profile behavior.
- Support the agent backends that have a stable per-run MCP configuration path:
  - `codex`
  - `claude`
  - `opencode`
- Fail fast with a short actionable error for backends that do not have a confirmed per-run MCP injection path in this change.

## Non-Goals

- Do not add a shared always-on MCP daemon across multiple agent sessions.
- Do not redesign `compare-result`, `compare-perf`, or `profile-report` as MCP tools in this change.
- Do not remove `run-command.py`.
- Do not change the public CLI subcommand names.
- Do not move run-eval implementation logic from `skills/triton-npu-run-eval/scripts/` into `src/triton_agent`.
- Do not add automatic NPU discovery.
- Do not silently fall back from MCP-backed execution to direct script execution when MCP injection is unavailable.

## Current Problem

Today the run-eval skill teaches agents to call:

```bash
python3 ./scripts/run-command.py <subcommand> ...
```

That has three problems for agent-backed workflows:

1. The top-level skill contract exposes script paths instead of stable capabilities.
2. NPU device control is distributed across agent prompts, direct script flags, and environment variables instead of being owned by one runtime boundary.
3. Backend-specific agent configuration has no generic way to declare per-request MCP servers, so run-eval cannot be surfaced as first-class tools.

## User-Facing Semantics

For agent-backed commands that stage `triton-npu-run-eval`, the agent should experience run-eval as four MCP tools instead of as shell commands.

Those commands currently include:

- `gen-eval`
- `gen-test`
- `gen-bench`
- `convert`
- `optimize`

The four MCP tools keep the existing command-level semantics for inputs, outputs, and artifacts:

- `run-test-baseline` still runs the generated test harness in baseline mode.
- `run-test-optimize` still runs the generated test harness in optimize mode.
- `run-bench` still produces the same benchmark perf artifact behavior.
- `profile-bench` still produces the same profile directory behavior.

The change is the invocation surface:

- agent-facing workflow: MCP tool call
- manual CLI and compatibility workflow: `run-command.py`

## Design Overview

The design has four layers:

1. A new skill-local FastMCP stdio server under `skills/triton-npu-run-eval/scripts/`
2. A generic request-time `mcp_servers` model on `AgentRequest`
3. Orchestration logic that attaches the run-eval MCP server when `triton-npu-run-eval` is staged
4. Backend-specific config writers that turn `mcp_servers` into concrete config files or CLI flags

## MCP Server Model

Add a backend-neutral MCP server definition model under `src/triton_agent` and store it on `AgentRequest` as:

- `mcp_servers`

This field is generic on purpose. The request model should not special-case run-eval because future commands may need additional MCP servers.

The neutral MCP server shape should include only what every supported backend needs for a local stdio server:

- server name
- transport type
- command
- args
- env

This change only needs stdio transport, but the request model should not hard-code run-eval-specific names or assumptions.

## When Requests Attach MCP Servers

Generation and optimize orchestration should attach the run-eval MCP server whenever:

- the staged skill set contains `triton-npu-run-eval`

That decision should remain backend-agnostic. The orchestration layer should not branch on agent type beyond constructing the shared `mcp_servers` definition.

If a request has no `mcp_servers`, backend behavior stays unchanged.

If a request has `mcp_servers` and the selected backend does not support request-scoped MCP injection in this design, the backend should fail before agent launch with a short actionable error.

## Run-Eval MCP Server

Add a new skill-local script:

- `skills/triton-npu-run-eval/scripts/run_eval_mcp_server.py`

Requirements:

- use `fastmcp`
- use stdio transport
- stay self-contained inside `skills/triton-npu-run-eval/scripts/`
- do not import `triton_agent`

The server should expose exactly these four tools in this phase:

- `run-test-baseline`
- `run-test-optimize`
- `run-bench`
- `profile-bench`

### Tool Naming

Use the existing command names directly as tool names.

That keeps:

- skill wording consistent
- trace and debugging terminology consistent
- user and maintainer mental models aligned with existing commands

### Tool Arguments

Tool arguments should stay aligned with the existing command arguments for the corresponding subcommand, except for NPU device selection.

The MCP tools should not expose `--npu-devices` or an equivalent per-call device-selection field.

The server owns device selection centrally.

### Tool Results

The MCP server should return structured JSON-like results instead of terminal-only text.

The result shape should preserve the information that the current command path already owns:

- `return_code`
- `stdout`
- `stderr`
- command-specific artifact path fields when present

Examples:

- test tools may return `archived_result`
- `run-bench` may return `perf_path`
- `profile-bench` may return `profile_dir`

The goal is semantic equivalence with the current command behavior while making the result machine-readable for MCP clients.

## NPU Resource Model

The server should accept these startup options:

- `--assigned-npus`
- `--workers-per-npu`

The CLI should source them from the existing environment variables:

- `TRITON_AGENT_BATCH_NPU_DEVICES`
- `TRITON_AGENT_BATCH_WORKERS_PER_NPU`

These variables already define the current repository-wide NPU pool contract, so this change should reuse that contract instead of inventing a second naming scheme.

### Startup Validation

If a request needs the run-eval MCP server and `TRITON_AGENT_BATCH_NPU_DEVICES` is unset or invalid, request construction should fail with a short actionable error.

If `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is unset, the current default of `1` should apply.

If it is set but invalid, request construction should fail with a short actionable error.

### Slot Pool Instead Of Global Semaphore Only

The server should model capacity as concrete execution slots, not just as one count.

Example:

- `assigned_npus=0,1`
- `workers_per_npu=2`

produces this slot pool:

- `0`
- `0`
- `1`
- `1`

Each tool invocation:

1. acquires one slot
2. receives one concrete device id
3. runs with that device
4. releases the slot in a `finally` path

This preserves two required behaviors:

- total concurrency is bounded by `len(assigned_npus) * workers_per_npu`
- execution still knows which concrete device it owns

### Environment Injection

The leased device id should be applied as:

- `ASCEND_RT_VISIBLE_DEVICES=<leased-device>`

This environment variable should be injected at the actual execution boundary, not at agent launch time.

That means:

- local tool execution injects it into the local runner process
- remote tool execution injects it into the remote command environment

The agent process itself should no longer receive device-control environment variables for run-eval behavior.

## Local And Remote Execution

The existing run-eval commands support both local and remote execution. The MCP-backed design should preserve that behavior.

For local execution:

- the leased device id is injected into the local subprocess environment

For remote execution:

- the leased device id is injected into the remote subprocess environment
- the server still owns the lease lifetime locally
- the remote host is assumed to interpret the leased device id in the same way the current CLI path does

This means the underlying skill-local helpers may need small signature extensions so all four tool paths can accept one leased device id consistently.

## Backend-Specific MCP Materialization

The backend layer should translate `AgentRequest.mcp_servers` into backend-specific runtime config under backend-owned workspace directories.

### Codex

Write project-scoped config to:

- `workdir/.codex/config.toml`

Use the existing Codex project config shape:

```toml
[mcp_servers.triton-agent-run-eval]
command = "python3"
args = ["..."]

[mcp_servers.triton-agent-run-eval.env]
KEY = "VALUE"
```

The backend should preserve unrelated existing Codex project config and only add or remove the managed MCP server entries it owns.

### Claude

Write the MCP config file to:

- `workdir/.claude/mcp.json`

Pass that file explicitly through:

- `--mcp-config <path>`

The file format should be the existing Claude MCP JSON shape:

```json
{
  "mcpServers": {
    "triton-agent-run-eval": {
      "type": "stdio",
      "command": "python3",
      "args": ["..."],
      "env": {
        "KEY": "VALUE"
      }
    }
  }
}
```

The backend should preserve unrelated files under `.claude/`.

### OpenCode

Extend the backend-owned config file:

- `workdir/.opencode/opencode.json`

Use OpenCode's local MCP shape under the `mcp` section:

- `type = "local"`
- `command = [...]`
- `environment = {...}`

This should merge with the temporary OpenCode config that the backend already stages for permission control.

### Unsupported Backends

In this change, the supported MCP-injection backends are:

- `codex`
- `claude`
- `opencode`

If `mcp_servers` is non-empty and the selected backend is:

- `pi`
- `traecli`
- `openhands`

the backend should fail before launch with a short explicit error instead of silently bypassing MCP.

## Managed Config Lifecycle

Backend-owned MCP config should be treated as runtime artifacts, not as persistent user-facing project state.

Requirements:

- preserve unrelated existing user configuration where the backend config file already exists
- add only the managed MCP server entries this run needs
- remove only the managed entries this run added during cleanup
- if the backend config file did not exist before and becomes empty after cleanup, remove it

If a managed server name already exists in a backend config file with a non-compatible user-owned definition, the backend should fail with a short actionable error instead of overwriting it silently.

## Skill Contract Changes

Update `skills/triton-npu-run-eval/SKILL.md` so its primary contract becomes:

- use the corresponding MCP tool for run-eval execution tasks

The skill should stop presenting `python3 ./scripts/run-command.py <subcommand> ...` as the normal agent-facing path.

`run-command.py` should remain documented only as:

- a compatibility path
- a manual CLI entrypoint
- a useful debugging path when patching run-eval helpers directly

## Compatibility Strategy

This design is intentionally additive.

It should preserve:

- the existing CLI commands
- the existing skill-local Python helpers
- the manual `run-command.py` workflow
- the existing artifact formats

What changes is the preferred execution surface for supported agent-backed workflows.

## Implementation Areas

The main code areas affected by this design are:

- `skills/triton-npu-run-eval/SKILL.md`
- `skills/triton-npu-run-eval/scripts/run_eval_mcp_server.py`
- skill-local run-eval helper modules that need one leased-device execution path
- `src/triton_agent/models.py`
- generation and optimize orchestration that build `AgentRequest`
- backend runners and backend config staging helpers for:
  - `codex`
  - `claude`
  - `opencode`

## Testing Strategy

### Request And Orchestration

Add tests that verify:

- requests only attach run-eval MCP servers when `triton-npu-run-eval` is staged
- `TRITON_AGENT_BATCH_NPU_DEVICES` is required for MCP-backed run-eval requests
- `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is parsed consistently with the existing batch contract
- the generated server command includes `--assigned-npus` and `--workers-per-npu`

### Backend Config Generation

Add backend tests that verify:

- Codex writes `workdir/.codex/config.toml` with the expected managed MCP entry
- Claude writes `workdir/.claude/mcp.json` and appends `--mcp-config <path>`
- OpenCode writes the expected `mcp` section into `.opencode/opencode.json`
- unrelated pre-existing backend config is preserved
- unsupported backends fail clearly when `mcp_servers` is non-empty

### MCP Server

Add tests that verify:

- the server exposes exactly the four required tools
- tool argument schemas stay aligned with the existing command semantics
- leased devices are assigned from the slot pool correctly
- devices are released after success and failure
- command-specific artifact paths are returned correctly

### Skill Contract

Add tests that verify:

- `triton-npu-run-eval/SKILL.md` instructs the agent to use MCP tools instead of direct script invocation as the primary workflow contract

## Verification

Implementation of this design should run the usual repository verification plus the skill-script checks required by repository policy.

At minimum:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

For touched skill-side Python files under `skills/*/scripts/`, also run:

- `bash scripts/run-skill-script-pyright.sh <path-to-each-touched-skill-script>`
