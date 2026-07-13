# Run-Bench Baseline Auto-Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--baseline-operator-file` to `run-bench` so the command can reuse or generate a baseline perf artifact and then automatically compare the candidate perf artifact against it.

**Architecture:** Keep benchmark execution semantics unchanged by treating baseline support as CLI-level orchestration around existing bench runner calls and the existing perf comparison helper. Touch both the top-level CLI and the skill-local run-command entrypoint so they stay aligned.

**Tech Stack:** Python, argparse, unittest, existing `run_local_bench` / `run_remote_bench` helpers, existing perf comparison helpers.

---

### Task 1: Add parser and handler regression tests for top-level `run-bench`

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_execution_commands.py`

- [ ] **Step 1: Add a parser test for `--baseline-operator-file`**

```python
def test_run_bench_accepts_baseline_operator_file(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-bench",
            "--bench-file",
            "bench_kernel.py",
            "--operator-file",
            "opt_kernel.py",
            "--baseline-operator-file",
            "kernel.py",
        ]
    )
    self.assertEqual(args.baseline_operator_file, "kernel.py")
```

- [ ] **Step 2: Run the CLI parser test to verify it fails**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k baseline_operator_file`
Expected: FAIL because `run-bench` parser does not expose `--baseline-operator-file`

- [ ] **Step 3: Add handler tests for baseline perf reuse and auto-compare**

```python
def test_handle_run_bench_reuses_existing_baseline_perf_and_auto_compares(self) -> None:
    ...

def test_handle_run_bench_generates_missing_baseline_perf_before_candidate_compare(self) -> None:
    ...
```

- [ ] **Step 4: Run the execution handler tests to verify they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_execution_commands.py -k run_bench`
Expected: FAIL because `handle_run_bench` neither accepts nor orchestrates baseline perf reuse/generation

### Task 2: Add skill-local and MCP regression tests

**Files:**
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_run_eval_mcp_server.py`
- Modify: `tests/test_run_eval_mcp_server_tool_metadata.py`

- [ ] **Step 1: Add a skill-local parser/help test for `--baseline-operator-file`**

```python
self.assertIn("--baseline-operator-file", completed.stdout)
```

- [ ] **Step 2: Add a skill-local dispatch test that reuses or generates baseline perf before compare**

```python
def test_script_run_bench_with_baseline_operator_auto_compares(self) -> None:
    ...
```

- [ ] **Step 3: Add MCP tests that `run-bench` exposes and forwards `baseline_operator_file`**

```python
self.assertEqual(
    tools["run-bench"].parameters["properties"]["baseline_operator_file"]["description"],
    "Optional absolute path to the baseline operator file used for automatic perf comparison.",
)
```

- [ ] **Step 4: Run the targeted tests to verify they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_command_script.py -k run_bench`
Expected: FAIL because the skill-local parser and dispatcher do not support baseline auto-compare

### Task 3: Implement top-level `run-bench` baseline orchestration

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/execution.py`

- [ ] **Step 1: Extend parser and path resolution**

```python
if spec.input_mode == "run-bench":
    subparser.add_argument("--bench-file", required=True)
    subparser.add_argument("--operator-file", required=True)
    subparser.add_argument("--baseline-operator-file")
    return
```

- [ ] **Step 2: Add a shared helper that derives baseline perf reuse/generation flow**

```python
def _derived_perf_path(operator_file: Path) -> Path:
    return operator_file.with_name(f"{operator_file.stem}_perf.txt")
```

- [ ] **Step 3: Update `handle_run_bench` to run baseline first only when needed, then compare**

```python
if baseline_operator_file is not None:
    baseline_perf_path = _derived_perf_path(baseline_operator_file)
    if not baseline_perf_path.exists():
        ...
    ...
    final_code = compare_perf_files(baseline_perf_path, perf_path)
```

- [ ] **Step 4: Run the top-level targeted tests to verify they pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py tests/test_execution_commands.py -k run_bench`
Expected: PASS

### Task 4: Implement skill-local and MCP support

**Files:**
- Modify: `skills/common/ascend-npu-run-eval/scripts/run-command.py`
- Modify: `src/helix/run_eval_mcp_server.py`

- [ ] **Step 1: Add `--baseline-operator-file` to the skill-local `run-bench` parser**

```python
run_bench.add_argument("--baseline-operator-file")
```

- [ ] **Step 2: Mirror the top-level orchestration in skill-local `run-bench` dispatch**

```python
baseline_operator_file = _resolve_optional_existing_path(
    parser, getattr(args, "baseline_operator_file", None), "Baseline operator file"
)
```

- [ ] **Step 3: Extend the MCP `run-bench` tool signature and argument forwarding**

```python
baseline_operator_file: Annotated[
    str | None,
    Field(description="Optional absolute path to the baseline operator file used for automatic perf comparison."),
] = None
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_command_script.py tests/test_run_eval_mcp_server.py tests/test_run_eval_mcp_server_tool_metadata.py -k run_bench`
Expected: PASS

### Task 5: Update docs and run final verification

**Files:**
- Modify: `README.md`
- Modify: `skills/common/ascend-npu-run-eval/references/run-bench.md`

- [ ] **Step 1: Document the new baseline auto-compare workflow**

```markdown
python3 ./scripts/run-command.py run-bench \
  --bench-file bench_<operator>.py \
  --operator-file opt_<operator>.py \
  --baseline-operator-file <operator>.py
```

- [ ] **Step 2: Run the skill-script strict pyright check**

Run: `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/run-command.py`
Expected: `0 errors`

- [ ] **Step 3: Run repository verification commands**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
Expected: PASS
