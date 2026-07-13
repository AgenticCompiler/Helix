# Run-Eval Shared HTTP MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-agent stdio run-eval MCP wiring with one shared per-execution HTTP MCP server, while keeping `AgentRequest` MCP references name-only and moving NPU slot management into the shared server.

**Architecture:** `src/helix/mcp.py` will own a nested process-local managed MCP scope that lazily starts a FastMCP HTTP server implemented under `src/helix/`. Backends will emit HTTP MCP config pointing at `http://127.0.0.1:<port>/mcp?workspace=<abs-path>`, skills will instruct agents to use MCP tools, and top-level request/batch/optimize entrypoints will wrap execution in a shared MCP scope so multiple agent runs reuse one server.

**Tech Stack:** Python 3, `fastmcp`, `uvicorn`, existing run-eval skill scripts, backend-local JSON/TOML config staging, `unittest`, `pytest`, `ruff`, `pyright`

---

## File Structure

- `docs/specs/2026-06-04-run-eval-mcp-server-design.md`
  - Shared HTTP design and lifecycle contract.
- `src/helix/mcp.py`
  - Managed MCP scope, shared HTTP server lifecycle, backend resolution helpers.
- `src/helix/run_eval_mcp_server.py`
  - FastMCP app and HTTP server runtime for the four run-eval tools.
- `src/helix/backends/base.py`
  - Unsupported backend fail-fast behavior remains centralized.
- `src/helix/backends/codex.py`
  - Emit `.codex/config.toml` HTTP MCP entries.
- `src/helix/backends/claude.py`
  - Emit `.claude/mcp.json` HTTP MCP entries and pass `--mcp-config`.
- `src/helix/backends/opencode.py`
  - Emit `.opencode/opencode.json` HTTP MCP entries using remote transport config.
- `src/helix/generation/orchestration.py`
  - Wrap request execution in a managed MCP scope.
- `src/helix/convert/orchestration.py`
  - Wrap request execution in a managed MCP scope.
- `src/helix/optimize/orchestration.py`
  - Wrap optimize execution in a managed MCP scope.
- `src/helix/generation/batch.py`
  - Use one outer managed MCP scope across all batch workspaces.
- `src/helix/convert/batch.py`
  - Use one outer managed MCP scope across all batch workspaces.
- `src/helix/optimize/batch.py`
  - Use one outer managed MCP scope across all batch workspaces.
- `skills/triton-npu-run-eval/SKILL.md`
  - MCP-first run-eval contract.
- `tests/test_codex_runner.py`
  - Expect URL-based MCP config.
- `tests/test_claude_runner.py`
  - Expect HTTP MCP config and `--mcp-config`.
- `tests/test_opencode_runner.py`
  - Expect remote HTTP MCP config.
- `tests/test_backends_base.py`
  - Preserve unsupported backend fail-fast coverage.
- `tests/test_generation_commands.py`
  - Keep request attachment coverage and add shared-scope orchestration coverage.
- `tests/test_convert_commands.py`
  - Add shared-scope and batch behavior coverage.
- `tests/test_optimize_runtime.py`
  - Add optimize shared-scope and batch behavior coverage.
- `tests/test_run_eval_mcp_server.py`
  - Move to the new `src` implementation and cover query-based workspace context.
- `tests/run_skill_test_utils.py`
  - Stop loading the old skill-local MCP server module once the new `src` module is the runtime implementation.

## Task 1: Rewrite Tests Around Shared HTTP Resolution

**Files:**
- Modify: `tests/test_codex_runner.py`
- Modify: `tests/test_claude_runner.py`
- Modify: `tests/test_opencode_runner.py`
- Modify: `tests/test_run_eval_mcp_server.py`

- [ ] **Step 1: Write failing expectations for HTTP-backed managed MCP config**

Update backend tests so they expect:

- Codex config contains:

```toml
[mcp_servers.helix-run-eval]
url = "http://127.0.0.1:<port>/mcp?workspace=<abs-path>"
```

- Claude config contains:

```json
{
  "mcpServers": {
    "helix-run-eval": {
      "type": "http",
      "url": "http://127.0.0.1:<port>/mcp?workspace=<abs-path>"
    }
  }
}
```

- OpenCode config contains:

```json
{
  "mcp": {
    "helix-run-eval": {
      "type": "remote",
      "url": "http://127.0.0.1:<port>/mcp?workspace=<abs-path>"
    }
  }
}
```

- [ ] **Step 2: Add failing server tests for query-based workspace parsing**

Add tests that verify the new run-eval MCP module:

- registers the same four tools
- rejects missing workspace query context
- passes the leased device into the compatibility runner

- [ ] **Step 3: Run focused tests and confirm they fail for the old stdio design**

Run: `uv run python -m unittest tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner tests.test_run_eval_mcp_server -v`

Expected: FAIL because the current implementation still emits stdio config and uses the skill-local server.

- [ ] **Step 4: Commit the test rewrite once it is green later**

```bash
git add tests/test_codex_runner.py tests/test_claude_runner.py tests/test_opencode_runner.py tests/test_run_eval_mcp_server.py
git commit -m "test: cover shared http run-eval mcp config"
```

## Task 2: Implement The Shared Managed MCP Scope

**Files:**
- Modify: `src/helix/mcp.py`
- Create: `src/helix/run_eval_mcp_server.py`

- [ ] **Step 1: Write failing scope tests**

Add tests for a process-local managed scope that:

- reuses one server instance inside nested contexts
- resolves backend entries only when a scope is active or can be created lazily
- emits URL-based config payloads with encoded absolute workspace paths

- [ ] **Step 2: Run the focused scope tests**

Run: `uv run python -m unittest tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner -v`

Expected: FAIL because `src/helix/mcp.py` still returns stdio launch commands.

- [ ] **Step 3: Implement the scope and HTTP resolver**

Implement in `src/helix/mcp.py`:

- `managed_mcp_server_names_for_skills(...)`
- a process-local shared scope context manager
- lazy scope creation helper
- backend-neutral managed server resolution that returns:

```python
{
    "helix-run-eval": {
        "transport": "http",
        "url": "http://127.0.0.1:8765/mcp?workspace=/abs/path"
    }
}
```

- [ ] **Step 4: Implement the FastMCP HTTP server module**

Create `src/helix/run_eval_mcp_server.py` with:

- FastMCP app creation
- query-string workspace parsing
- NPU slot pool creation from environment
- four tools matching the existing run-eval subcommands
- a background `uvicorn` runtime that can start on an ephemeral localhost port and shut down cleanly

- [ ] **Step 5: Re-run the focused scope/config tests**

Run: `uv run python -m unittest tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner tests.test_run_eval_mcp_server -v`

Expected: PASS

- [ ] **Step 6: Commit the shared scope implementation**

```bash
git add src/helix/mcp.py src/helix/run_eval_mcp_server.py tests/test_codex_runner.py tests/test_claude_runner.py tests/test_opencode_runner.py tests/test_run_eval_mcp_server.py
git commit -m "feat: add shared http run-eval mcp server"
```

## Task 3: Rewire Backends To Emit HTTP MCP Config

**Files:**
- Modify: `src/helix/backends/codex.py`
- Modify: `src/helix/backends/claude.py`
- Modify: `src/helix/backends/opencode.py`

- [ ] **Step 1: Write failing backend-specific assertions if still missing**

Make sure tests explicitly assert:

- Codex writes `url = ...` and no `command/args`
- Claude writes `type = http`
- OpenCode writes `type = remote`

- [ ] **Step 2: Run backend tests to confirm failure**

Run: `uv run python -m unittest tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner -v`

Expected: FAIL while config writers still assume stdio/local process launching.

- [ ] **Step 3: Implement backend config writers**

Update config writers to consume HTTP-managed server definitions and emit only the fields each backend needs.

- [ ] **Step 4: Re-run backend tests**

Run: `uv run python -m unittest tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner -v`

Expected: PASS

- [ ] **Step 5: Commit backend config changes**

```bash
git add src/helix/backends/codex.py src/helix/backends/claude.py src/helix/backends/opencode.py
git commit -m "feat: emit http mcp config for supported backends"
```

## Task 4: Share One MCP Scope Across Request And Batch Lifecycles

**Files:**
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/convert/orchestration.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/generation/batch.py`
- Modify: `src/helix/convert/batch.py`
- Modify: `src/helix/optimize/batch.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing lifecycle tests**

Add tests that verify:

- single request execution enters a managed MCP scope when `mcp_servers` is present
- optimize multi-invocation execution shares one scope across repeated runner calls
- batch execution shares one scope across multiple workspaces

Use a patched fake scope factory that records enter/exit counts and resolved URLs.

- [ ] **Step 2: Run focused lifecycle tests**

Run: `uv run python -m unittest tests.test_generation_commands tests.test_convert_commands tests.test_optimize_runtime -v`

Expected: FAIL because current orchestration does not create a shared managed MCP scope.

- [ ] **Step 3: Implement orchestration and batch scope wiring**

Wrap:

- `run_generation_request`
- `run_convert_request`
- `run_optimize_request`
- `run_gen_eval_batch`
- `run_convert_batch`
- `run_optimize_batch`

in the new managed MCP scope helper, while keeping nested scopes reusable instead of starting duplicate servers.

- [ ] **Step 4: Remove obsolete direct affinity injection for MCP-managed agent runs**

Update batch code so it no longer sets `ASCEND_RT_VISIBLE_DEVICES` on agent requests when MCP-managed run-eval is in play.

- [ ] **Step 5: Re-run lifecycle tests**

Run: `uv run python -m unittest tests.test_generation_commands tests.test_convert_commands tests.test_optimize_runtime -v`

Expected: PASS

- [ ] **Step 6: Commit lifecycle wiring**

```bash
git add src/helix/generation/orchestration.py src/helix/convert/orchestration.py src/helix/optimize/orchestration.py src/helix/generation/batch.py src/helix/convert/batch.py src/helix/optimize/batch.py tests/test_generation_commands.py tests/test_convert_commands.py tests/test_optimize_runtime.py
git commit -m "feat: share run-eval mcp scope across command execution"
```

## Task 5: Final Cleanup And Verification

**Files:**
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Modify: `tests/run_skill_test_utils.py`
- Delete or stop using: `skills/triton-npu-run-eval/scripts/run_eval_mcp_server.py`

- [ ] **Step 1: Align the skill and tests with the new runtime module**

Keep the skill MCP-first and stop treating the skill-local server script as the primary implementation surface.

- [ ] **Step 2: Run focused verification**

Run:

- `uv run python -m unittest tests.test_backends_base tests.test_generation_commands tests.test_convert_commands tests.test_optimize_runtime tests.test_codex_runner tests.test_claude_runner tests.test_opencode_runner tests.test_run_eval_mcp_server -v`

Expected: PASS

- [ ] **Step 3: Run required repository verification**

Run:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: PASS

- [ ] **Step 4: If any skill-side Python helper changed, run strict skill-script pyright**

Run only if a `skills/*/scripts/*.py` file changed:

- `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py`

- [ ] **Step 5: Commit cleanup and verification**

```bash
git add docs/specs/2026-06-04-run-eval-mcp-server-design.md docs/plans/2026-06-04-run-eval-mcp-server.md skills/triton-npu-run-eval/SKILL.md tests/run_skill_test_utils.py
git commit -m "docs: finalize shared http run-eval mcp design"
```
