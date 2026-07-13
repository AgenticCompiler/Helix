# NPU Operator Accuracy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old generic result-tolerance flow with the new NPU-only accuracy comparison contract across standalone tests, differential payloads, runner orchestration, CLI entrypoints, and generation specs.

**Architecture:** Keep the comparison authority in a shared skill-side runtime module, route both standalone and differential validation through that module, and simplify the CLI/skill wrappers so they no longer expose legacy compare-level controls. Generated test specs must describe the new module contract directly so agent-produced tests execute through the runner rather than by self-running Python files.

**Tech Stack:** Python, argparse, unittest, pyright, repository Markdown docs

---

### Task 1: Lock The New Comparison Contract In Focused Tests

**Files:**
- Modify: `tests/test_npu_compare.py`
- Modify: `tests/test_test_runner.py`
- Modify: `tests/run_skill_test_utils.py`

- [ ] **Step 1: Update focused tests to the new standalone and differential contract**

Replace legacy assumptions about `run_streaming_process`, legacy archived payloads, and standalone self-execution with assertions that match:
- `main(operator_api)` import-only standalone execution
- `{"compute": <bool>, "cases": [...]}` differential payloads
- remote helper scripts copying `npu_compare.py`
- direct warning filtering through the new result-filter helper

- [ ] **Step 2: Run the focused test suite and confirm the new red/green cycle**

Run:

```bash
uv run python -m unittest tests.test_npu_compare tests.test_test_runner -v
```

Expected: red before implementation alignment, then green once the new contracts are wired correctly.

### Task 2: Stabilize The Shared Comparison Runtime And Runner

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/npu_compare.py`
- Modify: `skills/triton-npu-run-eval/scripts/compare_result.py`
- Modify: `skills/triton-npu-run-eval/scripts/test_runner.py`

- [ ] **Step 1: Finish the shared comparison runtime**

Keep the five-path decision matrix, detailed diagnostics, explicit legacy-payload rejection, and shared artifact/case comparison entrypoints. Tighten typing so the file passes the required strict pyright check.

- [ ] **Step 2: Finish runner orchestration**

Keep standalone execution import-only, pass `operator_api` into `main(operator_api)`, archive differential inputs/results with the file-level compute flag, and ensure remote runner scripts mirror the local behavior.

- [ ] **Step 3: Run focused tests and skill-script pyright checks**

Run:

```bash
uv run python -m unittest tests.test_npu_compare tests.test_test_runner -v
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/npu_compare.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/test_runner.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/compare_result.py
```

Expected: PASS for all commands.

### Task 3: Remove Legacy Compare-Level Wiring From CLI And Command Surfaces

**Files:**
- Modify: `src/helix/commands/comparison.py`
- Modify: `src/helix/commands/execution.py`
- Modify: `src/helix/commands/convert.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/run_eval_mcp_server.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `tests/test_comparison_commands.py`
- Modify: `tests/test_execution_commands.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_run_eval_mcp_server_tool_metadata.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_remote_execution.py`

- [ ] **Step 1: Update parser and orchestration tests first**

Change tests so `compare-result` and differential `run-test` no longer expect `--compare-level`, default tolerance labels, or compare-level forwarding.

- [ ] **Step 2: Delete the legacy CLI plumbing**

Remove compare-level parser options, protocol arguments, handler validation, and forwarding code. Keep differential auto-compare behavior, but always use the new shared comparison authority.

- [ ] **Step 3: Run command/CLI tests**

Run:

```bash
uv run python -m unittest tests.test_comparison_commands tests.test_execution_commands tests.test_convert_commands tests.test_cli tests.test_run_eval_mcp_server_tool_metadata tests.test_skill_command_script tests.test_remote_execution -v
```

Expected: PASS.

### Task 4: Update Generation Specs, Skill Guidance, And User Docs

**Files:**
- Modify: `skills/triton-npu-gen-test/SKILL.md`
- Modify: `skills/triton-npu-gen-test/references/test-standalone-spec.md`
- Modify: `skills/triton-npu-gen-test/references/test-differential-spec.md`
- Modify: `skills/triton-npu-run-eval/references/run-test.md`
- Modify: `skills/triton-npu-run-eval/references/compare-result.md`
- Modify: `README.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Update generation and run-eval contracts**

Document:
- `# compute-kind: compute|non-compute`
- standalone `main(operator_api)` with no self-executing `__main__` block
- shared `compare_case_result(...)` usage instead of `assert_close`
- differential cases carrying `id`, `inputs`, and `fn`
- manual and automatic compare flows without compare-level choices

- [ ] **Step 2: Run documentation contract tests**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: PASS.

### Task 5: Run Repository Verification For The Changed Surfaces

**Files:**
- Verify only

- [ ] **Step 1: Run repository-standard verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS, or a precise report of any remaining unrelated failures.
