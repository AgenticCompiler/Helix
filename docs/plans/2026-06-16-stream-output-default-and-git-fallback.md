# Stream Output Default And Git Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the CLI to default to streamed agent output, rename the opt-out flag to `--no-stream-output`, reduce optimize round batch default to `5`, and skip temporary git initialization when `git` is unavailable.

**Architecture:** Keep runtime request semantics on the existing `show_output` boolean so backend and logging code stay stable. Limit behavior changes to CLI parsing/default mapping, optimize defaults, skill staging git detection, and user-facing docs/tests.

**Tech Stack:** Python, argparse, unittest, README docs

---

### Task 1: Lock The New CLI Contract In Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_report_command.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing parser and default-behavior assertions**

Add assertions covering:

```python
args = parser.parse_args(["gen-test", "-i", "kernel.py"])
self.assertTrue(args.show_output)

args = parser.parse_args(["gen-test", "-i", "kernel.py", "--no-stream-output"])
self.assertFalse(args.show_output)

with self.assertRaises(SystemExit):
    parser.parse_args(["gen-test", "-i", "kernel.py", "--show-output"])
```

- [ ] **Step 2: Run the focused CLI tests to verify they fail for the expected reason**

Run: `uv run python -m unittest tests.test_cli tests.test_report_command tests.test_convert_commands tests.test_models -v`

Expected: FAIL because the parser still exposes `--show-output`, defaults `show_output` to `False`, and request defaults still use round batch size `10`.

### Task 2: Add Missing-Git Skill Staging Regression Coverage

**Files:**
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Write the failing missing-git staging test**

Add coverage similar to:

```python
with mock.patch("triton_agent.skills.shutil.which", return_value=None):
    links = manager.prepare_skills("codex", workspace, skill_names=("triton-npu-gen-test",))
```

Then assert:

```python
self.assertIsNone(links.temporary_git_dir)
self.assertFalse((workspace / ".git").exists())
self.assertTrue((target / "triton-npu-gen-test" / "SKILL.md").exists())
```

- [ ] **Step 2: Run the focused skill tests to verify they fail first**

Run: `uv run python -m unittest tests.test_skills -v`

Expected: FAIL because the current implementation still calls `git init`.

### Task 3: Implement CLI And Default Changes

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/generation.py`
- Modify: `src/triton_agent/commands/convert.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/commands/report.py`
- Modify: `src/triton_agent/commands/report_batch.py`
- Modify: `src/triton_agent/commands/log_check.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/optimize/prompts.py`

- [ ] **Step 1: Rename the CLI flag and flip the default mapping**

Use `argparse` to add:

```python
subparser.add_argument(
    "--no-stream-output",
    dest="show_output",
    action="store_false",
    default=True,
)
```

and remove `--show-output` from the parser surface.

- [ ] **Step 2: Update command option builders to preserve the new parser default**

Keep code simple:

```python
show_output=bool(getattr(args, "show_output", True))
```

- [ ] **Step 3: Change round batch defaults from `10` to `5`**

Update the CLI option plus model defaults:

```python
subparser.add_argument("--round-batch-size", type=int, default=5)
round_batch_size = 99 if interact else getattr(args, "round_batch_size", 5)
round_batch_size: int = 5
```

- [ ] **Step 4: Run the focused CLI and command tests**

Run: `uv run python -m unittest tests.test_cli tests.test_report_command tests.test_convert_commands tests.test_models tests.test_optimize_commands tests.test_generation_commands -v`

Expected: PASS for the renamed flag, default streaming behavior, and round batch size `5`.

### Task 4: Implement Missing-Git Fallback And Refresh Docs

**Files:**
- Modify: `src/triton_agent/skills.py`
- Modify: `README.md`

- [ ] **Step 1: Guard temporary git initialization behind a git executable check**

Use logic like:

```python
if shutil.which("git") is None:
    return None
```

before `subprocess.run(["git", "init", "-q"], ...)`.

- [ ] **Step 2: Update README examples and shared-option docs**

Replace `--show-output` with `--no-stream-output` and describe streaming as the default behavior.

- [ ] **Step 3: Run focused docs-adjacent tests plus standard repo verification**

Run: `uv run python -m unittest tests.test_skills tests.test_cli tests.test_report_command tests.test_convert_commands -v`

Run: `uv run --group dev ruff check`

Run: `uv run pyright`

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: all commands exit `0`.
