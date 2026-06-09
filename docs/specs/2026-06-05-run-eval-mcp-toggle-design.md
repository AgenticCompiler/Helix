# Run-Eval MCP Toggle Design

## Summary

Add an explicit `--enable-mcp` toggle for agent-backed run-eval workflows so `triton-agent` can switch between:

- the existing script-oriented `triton-npu-run-eval` skill
- a new MCP-oriented skill source staged under the same workspace-visible skill name

When MCP is disabled, the current run-eval behavior and batch NPU concurrency checks remain unchanged. When MCP is enabled, the staged skill name remains `triton-npu-run-eval`, but its staged content comes from `skills/triton-npu-run-eval-mcp/`, agent backends receive MCP server config, and shared HTTP MCP server slot management becomes the authority for NPU-bound run-eval execution.

## Goals

- Add `--enable-mcp` only to agent-backed commands that stage the run-eval skill.
- Keep the staged workspace-visible skill name as `triton-npu-run-eval` in both modes.
- Preserve the existing non-MCP script workflow when `--enable-mcp` is not passed.
- Route MCP-enabled flows through the shared HTTP run-eval MCP server.
- Update the MCP skill content so it teaches tool usage instead of script usage.
- Add `profile-report` and `compare-perf` MCP tools.
- Remove `compare-result` from the MCP skill and MCP server while preserving the old CLI and script path.

## Non-Goals

- No change to local direct-execution subcommands such as `run-test`, `run-bench`, `compare-result`, `compare-perf`, or `report`.
- No removal of the legacy `triton-npu-run-eval` skill or its script entrypoints.
- No support for `compare-result` inside the MCP skill or MCP server.
- No cross-process shared MCP server design beyond the existing single-process shared HTTP scope.

## User-Visible Semantics

### Commands

Add `--enable-mcp` to exactly these commands:

- `gen-eval`
- `gen-eval-batch`
- `convert`
- `convert-batch`
- `optimize`
- `optimize-batch`

No other commands accept the toggle.

### Skill behavior

Without `--enable-mcp`:

- stage the existing `skills/triton-npu-run-eval/` content
- do not configure request-scoped MCP servers
- keep the script-oriented run-eval workflow unchanged

With `--enable-mcp`:

- stage a workspace-visible skill named `triton-npu-run-eval`
- source its staged content from `skills/triton-npu-run-eval-mcp/`
- configure the managed run-eval MCP server for supported backends
- teach the agent to use MCP tools instead of calling `python3 ./scripts/run-command.py`

### Batch concurrency

Without `--enable-mcp`:

- keep the existing batch NPU capacity validation
- continue relying on the existing batch/device affinity control path

With `--enable-mcp`:

- `--concurrency` limits only how many agent tasks run at once
- batch mode no longer requires `concurrency <= effective NPU slot capacity`
- NPU contention is resolved inside the shared MCP server by its slot pool

This means agent concurrency and NPU execution concurrency are intentionally decoupled in MCP mode.

## Architecture

### 1. Explicit MCP mode in command options

Add `enable_mcp: bool = False` to:

- `GenerationOptions`
- `ConvertOptions`
- `OptimizeRunOptions`
- `AgentRequest`

The CLI layer parses `--enable-mcp` for the six supported commands and propagates the value through the existing request-building path.

`AgentRequest.enable_mcp` becomes the runtime authority for whether a given agent invocation should receive MCP server configuration.

### 2. Stable staged skill name with source switching

Keep `resolve_staged_skills(...)` responsible for returning workspace-visible staged skill names and source overrides.

For run-eval-capable agent workflows:

- always include staged skill name `triton-npu-run-eval`
- when `enable_mcp` is false, do not override its source
- when `enable_mcp` is true, override its source to `triton-npu-run-eval-mcp`

This preserves one stable staged skill identity for prompt guidance, memory files, and agent-side behavior while allowing the CLI to choose which real repository directory is copied into the workspace.

### 3. MCP server activation

Managed MCP server activation should require both:

- staged run-eval skill presence
- `enable_mcp == true`

The current helper that derives managed MCP server names from staged skills should be updated so MCP activation is explicit rather than inferred only from staged skill names.

Result:

- non-MCP requests never attach `mcp_servers`
- MCP requests attach `("triton-agent-run-eval",)`

### 4. Batch control split

Batch command behavior should split by mode:

Non-MCP mode:

- preserve current `validate_batch_affinity_capacity(...)` behavior
- preserve current non-MCP run-eval resource assumptions

MCP mode:

- skip the batch-time `concurrency <= effective capacity` validation
- still use `ThreadPoolExecutor(max_workers=max_concurrency)` to control agent concurrency
- let the shared HTTP MCP server serialize or queue NPU-bound run-eval tool invocations according to its configured slot pool

This preserves batch-level backpressure on the number of agent sessions while moving NPU allocation authority into the MCP runtime.

### 5. MCP skill contract

The staged MCP skill must keep the name `triton-npu-run-eval` but change its content contract:

- instruct the agent to use MCP tools
- avoid normal-path script invocation guidance
- keep references focused on tool arguments, required artifacts, and expected outputs

The MCP skill should cover only:

- `run-test-baseline`
- `run-test-optimize`
- `run-bench`
- `profile-bench`
- `profile-report`
- `compare-perf`

`compare-result` must be removed from:

- `skills/triton-npu-run-eval-mcp/SKILL.md`
- `skills/triton-npu-run-eval-mcp/references/`
- MCP server tool registration

The legacy `skills/triton-npu-run-eval/` skill remains unchanged and continues documenting script usage.

### 6. Shared HTTP MCP server tool surface

Extend `src/triton_agent/run_eval_mcp_server.py` so the MCP server exposes:

- `run-test-baseline`
- `run-test-optimize`
- `run-bench`
- `profile-bench`
- `profile-report`
- `compare-perf`

NPU slot leasing rules:

- lease a device for `run-test-baseline`
- lease a device for `run-test-optimize`
- lease a device for `run-bench`
- lease a device for `profile-bench`
- do not lease a device for `profile-report`
- do not lease a device for `compare-perf`

This keeps the slot pool focused on device-bound execution instead of artifact-only analysis.

## Data Flow

### Non-MCP path

1. User runs an agent-backed command without `--enable-mcp`.
2. CLI builds options with `enable_mcp = false`.
3. Skill staging copies the existing `triton-npu-run-eval` skill.
4. Request building omits `mcp_servers`.
5. Batch mode continues enforcing existing NPU capacity validation.
6. Agent follows the script-oriented run-eval skill behavior.

### MCP path

1. User runs an agent-backed command with `--enable-mcp`.
2. CLI builds options with `enable_mcp = true`.
3. Skill staging stages `triton-npu-run-eval` from source `triton-npu-run-eval-mcp`.
4. Request building attaches the managed run-eval MCP server name.
5. Supported backends emit MCP configuration pointing at the shared local HTTP server with the workspace query.
6. Agent follows the MCP-oriented run-eval skill and calls MCP tools.
7. The MCP server leases NPU slots only for device-bound tools and queues excess requests.

## Error Handling

- Unsupported backends must continue failing fast when a request carries `mcp_servers`.
- The CLI must not silently enable MCP for commands that do not accept `--enable-mcp`.
- In MCP mode, missing or invalid NPU pool environment variables should continue to fail at managed MCP server startup with clear errors.
- In non-MCP mode, existing batch affinity validation failures should remain unchanged.
- MCP requests must fail clearly if an unknown managed MCP server name is requested.

## Testing Strategy

Update or add tests for:

- CLI parsing for `--enable-mcp` on the six supported commands
- rejection or absence of `--enable-mcp` on unrelated commands
- `resolve_staged_skills(...)` returning the stable staged name and the correct source override
- request builders attaching `enable_mcp` and `mcp_servers` only in MCP mode
- batch commands preserving old capacity validation in non-MCP mode
- batch commands skipping capacity validation in MCP mode
- MCP server tool registration now including `profile-report` and `compare-perf`
- MCP server tool execution for `profile-report` and `compare-perf` without device leasing
- absence of `compare-result` from MCP skill docs and MCP server tool registration
- backend config emission remaining correct for MCP-enabled requests

## Migration Notes

- Keep the legacy run-eval skill and script path as the default behavior.
- Treat the MCP skill as an alternate source for the same staged skill name rather than as a second user-facing skill identity.
- Preserve old `compare-result` behavior outside the MCP path.
