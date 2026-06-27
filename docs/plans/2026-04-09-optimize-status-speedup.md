# Optimize Status Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add geomean and total speedup reporting to `optimize-status`, switch best-round selection to geomean speedup, and align optimize workflow records with the new metric semantics.

**Architecture:** Keep perf files as the numeric source of truth. Extend the optimize status model and calculation helpers with two speedup metrics, render them in the CLI output, and update optimize guidance plus `opt-note.md` format docs so records and CLI summaries describe the same best-round logic.

**Tech Stack:** Python 3.11, `math`, existing optimize status/render helpers, Markdown guidance docs, Python `unittest`

---

### Task 1: Lock Status Metric Semantics In Tests

**Files:**
- Modify: `tests/test_optimize_status.py`
- Modify: `src/triton_agent/optimize/status.py`
- Modify: `src/triton_agent/optimize/models.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- geomean speedup and total speedup to be computed for a comparable round
- best round selection to prefer the highest geomean speedup

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_status -v`
Expected: FAIL because speedup fields do not exist yet and best-round selection still uses average improvement

- [ ] **Step 3: Write minimal implementation**

Extend the models and status calculation logic with the two speedup metrics and geomean-first ranking.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_status -v`
Expected: PASS

### Task 2: Lock CLI And Render Output

**Files:**
- Modify: `tests/test_optimize_render.py`
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/optimize/render.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- `optimize-status` render output to include `Geomean speedup` and `Total speedup`
- unknown states to render those fields as `unknown`

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_render tests.test_cli.PathResolutionTests.test_main_optimize_status_reports_numeric_best_and_logged_best -v`
Expected: FAIL because the renderer does not print the new metrics

- [ ] **Step 3: Write minimal implementation**

Render the new metrics without changing the TTY color strategy already in place.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_render tests.test_cli -v`
Expected: PASS

### Task 3: Align Optimize Workflow Records

**Files:**
- Modify: `skills/triton/triton-npu-optimize/references/opt-note-format.md`
- Modify: `skills/triton/triton-npu-optimize/references/workflow.md`
- Modify: `src/triton_agent/optimize/guidance.py`
- Modify: `docs/notes/2026-04-07-optimize-status-subcommand.md`

- [ ] **Step 1: Update `opt-note.md` format guidance**

Document the new overall-summary fields and state that the final best round is judged by geomean speedup.

- [ ] **Step 2: Update optimize workflow and guidance wording**

Tell optimizing agents to record geomean and total speedup in the overall summary and use geomean speedup as the best-round headline metric.

- [ ] **Step 3: Re-read status design docs**

Ensure the user-facing design document matches the new formulas and best-round semantics.

### Task 4: Verify End To End

**Files:**
- Modify: none

- [ ] **Step 1: Run focused verification**

Run: `uv run python -m unittest tests.test_optimize_status tests.test_optimize_render tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run repo verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
