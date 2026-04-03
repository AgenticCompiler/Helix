# Optimize Continue Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `optimize --continue` mode that resumes an existing optimization workspace, reuses existing harness modes, and fails fast when the expected session state is missing.

**Architecture:** Keep continue-mode detection in the CLI orchestration layer. The parser records whether optimize is a fresh run or a continue run, the CLI validates existing session artifacts and resolves harness metadata when continuing, and prompt generation adds explicit continuation wording without changing the optimize skill contract itself.

**Tech Stack:** Python `argparse`, existing metadata parsers, Python `unittest`

---

### Task 1: Lock Parser And Prompt Expectations

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add tests for:
- `optimize --continue` parsing into a dedicated destination
- prompt wording for continue mode
- no parser-level optimize mode defaults in continue-sensitive paths

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_cli.PromptTests -v`
Expected: FAIL because `--continue` is not implemented yet

- [ ] **Step 3: Write minimal implementation**

Update the parser and prompt builder until the new tests pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_cli.PromptTests -v`
Expected: PASS

### Task 2: Lock Continue-Mode Validation In Tests

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add tests for:
- rejecting `--continue --test-mode`
- rejecting `--continue --bench-mode`
- rejecting missing `opt-note.md`
- rejecting missing `opt-round-*`
- resolving modes from existing harness metadata in continue mode

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: FAIL because continue-mode validation is not implemented

- [ ] **Step 3: Write minimal implementation**

Add CLI helpers for continue validation and metadata resolution until the tests pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: PASS

### Task 3: Implement Continue Mode In CLI And Request Flow

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/models.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Use the tests from Tasks 1 and 2 as the red step.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: FAIL before implementation

- [ ] **Step 3: Write minimal implementation**

Implement:
- parser support for `--continue`
- explicit optimize mode resolution in `main()`
- continue-mode prompt wording
- documentation updates

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

### Task 4: Final Verification

**Files:**
- Modify: none
- Test: repo-wide checks

- [ ] **Step 1: Run focused tests**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
