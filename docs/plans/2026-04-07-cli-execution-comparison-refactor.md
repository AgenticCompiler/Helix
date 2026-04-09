# CLI Execution And Comparison Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move execution and comparison command handling out of `src/triton_agent/cli.py` while keeping command behavior unchanged.

**Architecture:** Add `commands/execution.py` and `commands/comparison.py` as the CLI-facing entrypoints for the remaining non-generation commands, and add small runtime modules to hold the non-parser execution and comparison helpers. Keep generation commands in `cli.py` for this phase so the migration stays focused.

**Tech Stack:** Python, `argparse`, `unittest`, existing run-skill wrapper modules

---

### Task 1: Add failing tests for the new execution and comparison module boundaries

**Files:**
- Create: `tests/test_execution_commands.py`
- Create: `tests/test_comparison_commands.py`

- [ ] **Step 1: Write focused tests that import the new modules directly**

Cover:
- execution metadata fallback helpers
- comparison handler validation and dispatch behavior

- [ ] **Step 2: Run the focused tests to verify they fail because the new modules do not exist yet**

Run: `uv run python -m unittest tests.test_execution_commands tests.test_comparison_commands -v`
Expected: FAIL with import errors or missing symbols

### Task 2: Extract execution runtime helpers and command handlers

**Files:**
- Create: `src/triton_agent/execution.py`
- Create: `src/triton_agent/commands/execution.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_execution_commands.py`

- [ ] **Step 1: Move run-test and run-bench helper behavior into `src/triton_agent/execution.py`**

Include:
- local and remote wrappers
- harness metadata fallback helpers

- [ ] **Step 2: Add `src/triton_agent/commands/execution.py` and move CLI-facing command handling there**

Keep:
- path validation
- error reporting
- result printing

- [ ] **Step 3: Run focused execution tests**

Run: `uv run python -m unittest tests.test_execution_commands -v`
Expected: PASS

### Task 3: Extract comparison runtime helpers and command handlers

**Files:**
- Create: `src/triton_agent/comparison.py`
- Create: `src/triton_agent/commands/comparison.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_comparison_commands.py`

- [ ] **Step 1: Move compare-result and compare-perf helper behavior into `src/triton_agent/comparison.py`**

- [ ] **Step 2: Add `src/triton_agent/commands/comparison.py` and move CLI-facing command handling there**

- [ ] **Step 3: Run focused comparison tests**

Run: `uv run python -m unittest tests.test_comparison_commands -v`
Expected: PASS

### Task 4: Re-run CLI and repository verification

**Files:**
- Modify: `src/triton_agent/cli.py`

- [ ] **Step 1: Run CLI regression tests**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
