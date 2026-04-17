# Inspect IR Ranking And Change Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `inspect_ir.py` so agents can rank stages by usefulness and scan adjacent stage transitions for the most significant changes.

**Architecture:** Keep the enhancement inside `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`. Reuse the existing stage discovery and keyword-counting helpers to compute a lightweight per-stage interestingness score and a lightweight adjacent-stage change score. Surface both through terminal-friendly commands: `list-stages --sort-by ...` and a new `find-changes` subcommand. Update the IR analyzer skill docs so these commands become the default navigation path for large archives.

**Tech Stack:** Python 3.11, `argparse`, `difflib`, `pathlib`, `unittest`, Markdown docs

---

### Task 1: Add failing tests for ranking and adjacent change scanning

**Files:**
- Modify: `tests/test_inspect_ir.py`

- [ ] **Step 1: Add failing tests for `list-stages --sort-by size`**
- [ ] **Step 2: Add failing tests for `list-stages --sort-by interesting`, including score display**
- [ ] **Step 3: Add failing tests for a new `find-changes` text renderer**
- [ ] **Step 4: Run the targeted inspect tests and confirm the new behavior is missing**

### Task 2: Implement scoring and `find-changes`

**Files:**
- Modify: `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`

- [ ] **Step 1: Add per-stage keyword-count and interestingness helpers**
- [ ] **Step 2: Extend `list-stages` sorting and rendering to show scores when useful**
- [ ] **Step 3: Add adjacent-stage change scoring and keyword delta summaries**
- [ ] **Step 4: Add parser wiring and output for `find-changes`**
- [ ] **Step 5: Run targeted inspect tests and make them pass**

### Task 3: Update skill and docs

**Files:**
- Modify: `skills/triton-npu-analyze-ir/SKILL.md`
- Modify: `README.md`
- Modify: `docs/2026-04-08-inspect-ir-ranking-and-change-scan.md`

- [ ] **Step 1: Update the skill so ranking and change scanning are recommended before ad hoc file browsing**
- [ ] **Step 2: Add one concise README example for `find-changes`**
- [ ] **Step 3: Refine the design doc if implementation details change visible behavior**

### Task 4: Verify the change

**Files:**
- Modify: `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`
- Modify: `tests/test_inspect_ir.py`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_inspect_ir -v`**
- [ ] **Step 2: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 3: Run `uv run pyright`**
- [ ] **Step 4: Run `uv run --group dev ruff check skills/triton-npu-analyze-ir/scripts/inspect_ir.py tests/test_inspect_ir.py`**
- [ ] **Step 5: Fix regressions and re-run verification until clean**
