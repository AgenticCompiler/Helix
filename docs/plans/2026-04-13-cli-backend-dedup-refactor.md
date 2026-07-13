# CLI And Backend Dedup Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant code in the CLI entrypoint and backend runners while preserving all current CLI behavior.

**Architecture:** Convert the CLI entrypoint into a thin executable module backed by table-driven command definitions, and centralize shared backend runner flow in the backend base class. Keep dedicated modules responsible for execution, comparison, generation, optimize, and backend-specific command assembly.

**Tech Stack:** Python, argparse, unittest, ruff, pyright, uv

---

### Task 1: Lock In Direct Module Imports Instead Of CLI Passthroughs

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_comparison_commands.py`
- Modify: `tests/test_execution_commands.py`

- [ ] **Step 1: Update tests that import helper functions from `helix.cli` to import from the dedicated modules instead**
- [ ] **Step 2: Run focused CLI-related tests and confirm they fail only where the old passthrough surface is still assumed**
- [ ] **Step 3: Keep parser and command-dispatch coverage intact so the CLI entrypoint still has regression protection**

### Task 2: Remove Redundant CLI Wrapper Surface

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/generation/__init__.py`
- Modify: `src/helix/output.py`
- Modify: any tests identified in Task 1

- [ ] **Step 1: Delete redundant passthrough helpers from `cli.py` and keep only entrypoint responsibilities**
- [ ] **Step 2: Ensure tests and internal imports use the real modules directly**
- [ ] **Step 3: Re-run focused tests to confirm no behavior change from the API surface cleanup**

### Task 3: Make CLI Parsing And Dispatch Table-Driven

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add an internal command-definition structure that describes handlers and supported options per command**
- [ ] **Step 2: Refactor `build_parser()` to construct subparsers from shared helpers driven by that structure**
- [ ] **Step 3: Refactor `main()` to dispatch via the same definitions instead of a long `if` chain**
- [ ] **Step 4: Run CLI tests and fix only parser or dispatch regressions introduced by the refactor**

### Task 4: Centralize Shared Backend Runner Flow

**Files:**
- Modify: `src/helix/backends/base.py`
- Modify: `src/helix/backends/codex.py`
- Modify: `src/helix/backends/opencode.py`
- Modify: `src/helix/backends/pi.py`
- Modify: `src/helix/backends/claude.py`
- Modify: backend runner tests as needed

- [ ] **Step 1: Write or update tests that rely on shared runner behavior staying stable**
- [ ] **Step 2: Move shared `run()`, `resume()`, mode selection, and verbose logging into the base runner**
- [ ] **Step 3: Keep per-backend modules focused on command construction and true backend-specific hooks**
- [ ] **Step 4: Run focused backend tests and make the minimal compatibility fixes needed**

### Task 5: Verify The Full Refactor

**Files:**
- Modify: `docs/specs/2026-04-13-cli-backend-dedup-refactor-design.md`
- Modify: `docs/plans/2026-04-13-cli-backend-dedup-refactor.md`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_cli tests.test_execution_commands tests.test_comparison_commands tests.test_generation_commands tests.test_backends_factory tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner -v`**
- [ ] **Step 2: Run `uv run --group dev ruff check`**
- [ ] **Step 3: Run `uv run pyright`**
- [ ] **Step 4: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 5: If verification fails, fix only regressions caused by this refactor and re-run the affected checks**
