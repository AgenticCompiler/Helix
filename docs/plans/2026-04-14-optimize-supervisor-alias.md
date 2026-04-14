# Optimize Supervisor Alias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--supervisor` as a compatibility alias for `--supervise on|off` on optimize commands without changing runtime behavior.

**Architecture:** Keep the existing `args.supervise` field as the single source of truth. Extend `argparse` registration so both flag names feed the same destination, then verify alias parsing with focused CLI tests.

**Tech Stack:** Python `argparse`, `unittest`

---

### Task 1: Add parser coverage for the alias

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
def test_optimize_command_accepts_supervisor_alias(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--supervisor", "on"])
    self.assertEqual(args.supervise, "on")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.TestCli.test_optimize_command_accepts_supervisor_alias -v`
Expected: FAIL because `--supervisor` is not registered yet.

- [ ] **Step 3: Write minimal implementation**

Register `--supervisor` alongside `--supervise` with the same `choices`, default, and destination.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.TestCli.test_optimize_command_accepts_supervisor_alias -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py src/triton_agent/cli.py docs/specs/2026-04-14-optimize-supervisor-alias-design.md docs/plans/2026-04-14-optimize-supervisor-alias.md
git commit -m "feat: add optimize supervisor alias"
```

### Task 2: Extend batch coverage and verify no behavior drift

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
def test_optimize_batch_accepts_supervisor_alias(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize-batch", "-i", "kernels", "--supervisor", "off"])
    self.assertEqual(args.supervise, "off")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.TestCli.test_optimize_batch_accepts_supervisor_alias -v`
Expected: FAIL because `--supervisor` is not registered yet.

- [ ] **Step 3: Reuse the same minimal implementation**

Ensure the alias is added in the shared optimize-option registration so both optimize commands inherit it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.TestCli.test_optimize_command_accepts_supervisor_alias tests.test_cli.TestCli.test_optimize_batch_accepts_supervisor_alias -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py src/triton_agent/cli.py
git commit -m "test: cover optimize supervisor alias"
```
