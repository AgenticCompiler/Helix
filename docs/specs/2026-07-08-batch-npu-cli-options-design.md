# Batch NPU CLI Options Design

## Summary

Promote batch NPU affinity configuration from environment-only controls to
first-class CLI options while preserving the existing environment variables as a
compatibility fallback. The new options should become the primary user-facing
interface for batch commands and the standalone `run-eval-mcp-server` command.
Managed MCP startup should receive explicit configuration from parsed command
options instead of re-reading global environment state.

## Goal

- Add `--npu-devices` and `--workers-per-npu` as first-class CLI options for
  the batch-affinity feature set.
- Preserve `TRITON_AGENT_BATCH_NPU_DEVICES` and
  `TRITON_AGENT_BATCH_WORKERS_PER_NPU` as compatibility inputs during parsing.
- Make parsed CLI values the source of truth for batch command execution and
  managed MCP startup.
- Keep the standalone `run-eval-mcp-server` executable behavior consistent with
  other CLI commands by supporting both explicit options and environment
  fallback.
- Preserve the current managed MCP rule that `workers-per-npu` is accepted but
  ignored when building the runtime device pool.

## Non-Goals

- Removing the existing batch environment variables.
- Changing the existing `run-bench` or `probe-bench` user-facing
  `--npu-devices` flag name.
- Changing non-batch execution semantics for `run-bench`, `probe-bench`,
  `run-test`, or other single-workspace commands.
- Changing managed MCP device-leasing semantics so that
  `workers-per-npu` affects runtime parallelism.
- Introducing a repository-wide config file for batch affinity.

## Current Behavior

Today the batch-affinity feature is controlled directly by:

- `TRITON_AGENT_BATCH_NPU_DEVICES`
- `TRITON_AGENT_BATCH_WORKERS_PER_NPU`

Runtime modules read those environment variables directly:

- `src/triton_agent/batch/affinity.py` for batch command affinity,
  capacity checks, and `--concurrency max`
- `src/triton_agent/eval/mcp_server.py` for managed and standalone
  run-eval MCP device-pool startup

This creates two practical issues:

- batch-affinity configuration is harder to discover than ordinary CLI options
- managed MCP implicitly depends on process-global environment state rather than
  the parsed command configuration that triggered server startup

## User-Facing Semantics

Add two primary CLI options:

- `--npu-devices`
- `--workers-per-npu`

These options apply to the batch-affinity feature set:

- `gen-eval` when used in batch mode through `--concurrency`
- `gen-eval-batch`
- `convert` when used in batch mode through `--concurrency`
- `convert-batch`
- `optimize` when used in batch mode through `--concurrency`
- `optimize-batch`
- `run-eval-mcp-server`

The compatibility rule is:

1. If the user passes the CLI option, use it.
2. Otherwise, if the legacy environment variable is set, use that value.
3. Otherwise, leave the value unset and preserve current defaults.

Examples:

```bash
uv run triton-agent optimize-batch -i operators --concurrency max \
  --npu-devices 0,1 --workers-per-npu 2
```

```bash
export TRITON_AGENT_BATCH_NPU_DEVICES=0,1
export TRITON_AGENT_BATCH_WORKERS_PER_NPU=2
uv run triton-agent optimize-batch -i operators --concurrency max
```

```bash
export TRITON_AGENT_BATCH_NPU_DEVICES=0,1
uv run triton-agent optimize-batch -i operators --concurrency max \
  --npu-devices 3,4
```

In the third example, the explicit CLI device list wins.

## Design

### 1. Parser owns compatibility fallback

Normalize batch-affinity inputs in the CLI layer after argument parsing.

The parser should treat the new options as the primary interface and use the
legacy environment variables only as fallback. This keeps compatibility logic in
one place instead of spreading it across runtime modules.

Recommended normalized fields on `argparse.Namespace`:

- `npu_devices`
- `workers_per_npu`

Those names may be reused across subcommands. The meaning is still scoped by the
selected command.

### 2. Batch-affinity helpers become pure input-driven helpers

Refactor `src/triton_agent/batch/affinity.py` so that it no longer owns direct
environment lookup for batch command execution. It should keep:

- device-list parsing
- workers-per-npu parsing
- slot expansion
- effective capacity calculation
- `--concurrency max` resolution
- affinity pool creation

But those helpers should operate on explicit raw values passed from the command
layer instead of reading `os.environ` themselves for the normal batch path.

### 3. Managed MCP startup receives explicit config

When `--enable-mcp` causes `triton-agent` to auto-start the internal run-eval
MCP server, that startup path should receive the normalized CLI values
explicitly.

The explicit flow should be:

- command parses args
- parser resolves `option > env`
- command / request model carries normalized values
- managed MCP startup receives those values as parameters

This avoids hidden dependency on ambient process environment for managed MCP.

### 4. Standalone MCP server still supports environment fallback

The standalone `run-eval-mcp-server` command is still an executable user entry
point, so it should continue to support environment fallback.

The startup functions should therefore support both modes:

- explicit parameters when provided by the caller
- legacy environment fallback when parameters are omitted

That means `start_http_server(...)` and `serve_http_server_forever(...)` should
accept optional `npu_devices` and `workers_per_npu` arguments and resolve
fallback only when those arguments are not supplied.

### 5. Managed MCP keeps current workers-per-npu semantics

Do not change the current managed MCP rule:

- `workers-per-npu` remains accepted for compatibility
- managed MCP still leases one active tool invocation per physical configured
  device
- standalone and managed run-eval MCP device pools continue to ignore expanded
  worker-slot semantics

This preserves the behavior already documented in
`2026-06-08-run-eval-mcp-ignore-workers-per-npu-design.md`.

## Affected Runtime Paths

### Batch command path

For `gen-eval`, `convert`, and `optimize` batch flows:

- parse `--npu-devices`
- parse `--workers-per-npu`
- apply `option > env` fallback
- pass normalized values to batch-affinity helpers
- derive capacity, slot pool, and `ASCEND_RT_VISIBLE_DEVICES` assignment from
  those normalized values

### Managed MCP path

For agent-backed flows with `--enable-mcp`:

- carry normalized values into the request / managed MCP context
- start the internal HTTP MCP server with explicit values
- avoid direct batch-affinity environment lookup in the managed path

### Standalone MCP server path

For `run-eval-mcp-server`:

- parse `--npu-devices`
- parse `--workers-per-npu`
- apply `option > env` fallback
- pass normalized values into server startup
- if startup is invoked without explicit values, continue to support direct
  environment fallback for executable-style use

## Validation Rules

For batch commands:

- `--workers-per-npu` is only meaningful when `--npu-devices` is configured
- when devices are configured, `workers-per-npu` defaults to `1`
- invalid values fail before any workspace launch
- `--concurrency max` uses normalized batch-affinity inputs instead of raw
  environment state

For managed MCP:

- `--workers-per-npu` may still be parsed and validated
- runtime device-pool size still follows configured physical devices only

For standalone `run-eval-mcp-server`:

- explicit options override environment variables
- omitted options may still fall back to environment variables
- when neither options nor env provide devices, preserve the current default to
  device `0`

## Implementation Shape

The smallest clean implementation is:

1. Extend CLI subparsers with `--npu-devices` and `--workers-per-npu` where the
   batch-affinity feature applies.
2. Add one CLI normalization helper that applies `option > env`.
3. Thread normalized values into command option models and managed MCP startup.
4. Refactor batch-affinity helpers to accept explicit raw values for the batch
   execution path.
5. Refactor MCP server startup helpers to accept optional explicit values and
   use environment fallback only when omitted.

This keeps the compatibility boundary close to the CLI while preserving the
standalone executable behavior of the MCP server.

## Documentation Changes

Update user-facing docs to make CLI options the preferred interface while
documenting environment variables as compatibility fallback:

- CLI help text
- README environment-variable table
- README batch-affinity examples
- README run-eval MCP server section

The docs should clearly say:

- use `--npu-devices` and `--workers-per-npu` for new invocations
- legacy `TRITON_AGENT_BATCH_*` variables still work
- explicit options override legacy environment variables
- managed MCP still ignores workers-per-npu for runtime leasing

## Testing And Verification

Add or update tests for:

- parser fallback from env when CLI options are omitted
- explicit CLI options overriding legacy environment variables
- batch `--concurrency max` using normalized option values
- batch slot expansion using normalized option values
- managed MCP startup receiving explicit option-derived values
- standalone MCP server option parsing plus environment fallback
- managed MCP continuing to ignore workers-per-npu in runtime device leasing

Expected test areas:

- `tests/test_cli.py`
- `tests/test_npu_affinity.py`
- `tests/test_generation_batch.py`
- `tests/test_convert_commands.py`
- `tests/test_optimize_runtime.py`
- `tests/test_run_eval_mcp_server.py`
- backend tests that cover managed MCP config generation if needed

Repository verification should include:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

## Files Expected To Change

- `src/triton_agent/cli.py`
- `src/triton_agent/batch/affinity.py`
- `src/triton_agent/commands/generation.py`
- `src/triton_agent/commands/convert.py`
- `src/triton_agent/commands/optimize.py`
- `src/triton_agent/commands/mcp_server.py`
- `src/triton_agent/generation/models.py`
- `src/triton_agent/convert/models.py`
- `src/triton_agent/optimize/models.py`
- `src/triton_agent/models.py`
- `src/triton_agent/eval/mcp.py`
- `src/triton_agent/eval/mcp_server.py`
- `README.md`
- corresponding tests
