# Optimize Batch Post-Optimize Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--post-optimize-command` to `optimize-batch` so each successful workspace optimize run can trigger one local shell command inside that workspace.

**Architecture:** Keep the feature inside optimize batch orchestration. Parse the new option into `OptimizeRunOptions`, then have the batch success path execute the command before upload/report/status-finalization. A non-zero command exit turns that workspace into a normal batch failure without stopping sibling workspaces.

**Tech Stack:** Python, `argparse`, `subprocess`, `unittest`.

---

### Task 1: Add failing parser coverage

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/models.py`

- [ ] **Step 1: Write the failing parser test**

```python
def test_optimize_batch_accepts_post_optimize_command(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["optimize-batch", "-i", "kernels", "--post-optimize-command", "echo done"]
    )
    self.assertEqual(args.post_optimize_command, "echo done")
    options = optimize_run_options_from_args(args)
    self.assertEqual(options.post_optimize_command, "echo done")
```

- [ ] **Step 2: Run the parser test and verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_batch_accepts_post_optimize_command -v`
Expected: FAIL because the parser does not know `--post-optimize-command`.

- [ ] **Step 3: Add the minimal parser plumbing**

```python
post_optimize_command=getattr(args, "post_optimize_command", None)
```

- [ ] **Step 4: Re-run the parser test**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_batch_accepts_post_optimize_command -v`
Expected: PASS

### Task 2: Add failing batch runtime coverage

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/triton_agent/optimize/batch.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
def test_run_optimize_batch_executes_post_optimize_command_after_success(self) -> None:
    ...

def test_run_optimize_batch_marks_workspace_failed_when_post_optimize_command_fails(self) -> None:
    ...
```

- [ ] **Step 2: Run the focused runtime tests and verify they fail**

Run: `uv run python -m unittest tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_executes_post_optimize_command_after_success tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_marks_workspace_failed_when_post_optimize_command_fails -v`
Expected: FAIL because optimize batch does not execute a post command yet.

- [ ] **Step 3: Implement the batch helper and success-path hook**

```python
completed = run_post_optimize_command(options.post_optimize_command, item.workspace)
if completed.returncode != 0:
    ...
```

- [ ] **Step 4: Re-run the focused runtime tests**

Run: `uv run python -m unittest tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_executes_post_optimize_command_after_success tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_marks_workspace_failed_when_post_optimize_command_fails -v`
Expected: PASS

### Task 3: Update user-facing docs and run targeted verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the new batch option**

```text
uv run triton-agent optimize-batch --input operators_root --post-optimize-command "..."
```

- [ ] **Step 2: Run the focused verification set**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_batch_accepts_post_optimize_command tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_executes_post_optimize_command_after_success tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_marks_workspace_failed_when_post_optimize_command_fails -v`
Expected: PASS
