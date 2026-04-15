# Opt Note Overall Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make completed optimize sessions leave a concise final outcome in `opt-note.md` and let optimize-status reuse that conclusion when available.

**Architecture:** Keep the user-facing change centered on the existing `opt-note.md` artifact. Add parsing helpers in the local optimize-status code so the CLI understands the new final summary block, then update optimize guidance and optimize-skill references so future optimization runs produce and refresh that block consistently.

**Tech Stack:** Python 3.11, `pathlib`, `re`, existing optimize-status helpers, Python `unittest`

---

### Task 1: Lock Summary Parsing Semantics In Tests

**Files:**
- Modify: `tests/test_optimize_status.py`
- Test: `tests/test_optimize_status.py`

- [ ] **Step 1: Write the failing test**

Add tests that define:
- parsing `Final best round` from an `## Overall Summary` block
- falling back to the legacy `Best status: current best` markers when the summary is absent
- warning when the summary final-best round disagrees with the legacy logged best marker

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_status -v`
Expected: FAIL because summary-aware parsing does not exist yet

- [ ] **Step 3: Write minimal implementation**

Add the smallest parsing helpers needed for the new summary block while preserving legacy behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_status -v`
Expected: PASS

### Task 2: Lock CLI Rendering And Warning Behavior

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add an optimize-status test covering a workspace where:
- `opt-note.md` contains `## Overall Summary`
- the numeric best round matches the summary best round
- the old per-round `current best` marker disagrees

Assert that:
- `Logged best` reflects the final summary best round
- a warning reports the disagreement with the legacy marker

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliMainTests.test_main_optimize_status_reports_numeric_best_and_logged_best -v`
Expected: FAIL before parser integration

- [ ] **Step 3: Write minimal implementation**

Wire the summary-aware parsing into optimize-status result calculation and warning generation.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliMainTests.test_main_optimize_status_reports_numeric_best_and_logged_best -v`
Expected: PASS

### Task 3: Update Guidance And Reference Docs

**Files:**
- Modify: `src/triton_agent/optimize/guidance.py`
- Modify: `skills/optimize/references/opt-note-format.md`
- Modify: `skills/optimize/references/workflow.md`

- [ ] **Step 1: Update `opt-note.md` format guidance**

Document the final `## Overall Summary` block and clarify that it should be refreshed at session completion.

- [ ] **Step 2: Update optimize workflow guidance**

Tell optimizing agents to leave `opt-note.md` with both round history and one concise final conclusion.

- [ ] **Step 3: Update CLI optimize guidance**

Make the generated optimize instructions explicitly require a final overall summary so future runs follow the new format.

### Task 4: Final Verification

**Files:**
- Modify: none
- Test: repo-wide checks

- [ ] **Step 1: Run focused tests**

Run: `uv run python -m unittest tests.test_optimize_status -v`
Expected: PASS

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
