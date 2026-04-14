# Optimize Batch Default Max Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `optimize-batch` so `--max-concurrency` defaults to `1` instead of `2`, without changing `gen-eval-batch`.

**Architecture:** Keep the change localized to the CLI command spec for `CommandKind.OPTIMIZE_BATCH`. Update the parser-level regression test and README batch optimize docs so the documented default matches the executable behavior.

**Tech Stack:** Python, `argparse`, `unittest`, Markdown docs

---

### Task 1: Lock The New Default In Tests

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Change the `optimize-batch` parser default assertion from `2` to `1`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_batch_maps_to_command_kind`
Expected: FAIL because the parser still defaults `optimize-batch --max-concurrency` to `2`

- [ ] **Step 3: Write minimal implementation**

Update the `CommandKind.OPTIMIZE_BATCH` CLI spec in `src/triton_agent/cli.py` so `max_concurrency_default=1`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_optimize_batch_maps_to_command_kind`
Expected: PASS

### Task 2: Update Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the batch optimize docs**

Document that `optimize-batch` defaults `--max-concurrency` to `1`.

- [ ] **Step 2: Run focused regression tests**

Run: `uv run python -m unittest tests.test_cli`
Expected: PASS

- [ ] **Step 3: Run repo verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
