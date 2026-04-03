# Batch Optimize Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `optimize-batch` CLI subcommand that scans immediate child directories, auto-detects one operator file per workspace, and runs bounded parallel optimize workflows with a final summary.

**Architecture:** Keep the new behavior in the CLI orchestration layer. The parser exposes a batch-only command shape, a small helper resolves candidate operator files per workspace, and a bounded thread pool reuses the existing single-workspace optimize execution helper so backend behavior, skill staging, optimize guidance, and supervision remain aligned with ordinary `optimize`.

**Tech Stack:** Python 3.11, `argparse`, `concurrent.futures`, `unittest`, existing Triton agent CLI modules

---

### Task 1: Add parser coverage and batch selection tests

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing parser tests for `optimize-batch`**
- [ ] **Step 2: Run the targeted unittest cases and confirm they fail for missing command support**
- [ ] **Step 3: Write failing behavior tests for candidate selection, bounded concurrency orchestration, and summary exit codes**
- [ ] **Step 4: Run the targeted unittest cases and confirm the new behavior still fails before implementation**

### Task 2: Implement batch optimize orchestration

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/prompts.py`

- [ ] **Step 1: Add the new command kind and parser wiring**
- [ ] **Step 2: Extract a reusable single-workspace optimize execution helper from the existing CLI flow**
- [ ] **Step 3: Implement workspace scanning, candidate filtering, and bounded parallel execution**
- [ ] **Step 4: Add compact batch result rendering and non-zero aggregate failure behavior**
- [ ] **Step 5: Run targeted unittests and make them pass**

### Task 3: Update user-facing docs

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Create: `docs/2026-04-03-batch-optimize-subcommand.md`

- [ ] **Step 1: Document the new command examples and semantics in `README.md`**
- [ ] **Step 2: Update `AGENTS.md` so the new subcommand is part of the durable command contract**
- [ ] **Step 3: Re-read the design doc and make sure the implementation still matches it**

### Task 4: Verify the whole change

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 2: Run `uv run --group dev ruff check`**
- [ ] **Step 3: Run `uv run pyright`**
- [ ] **Step 4: Fix any regressions and re-run verification until clean**
