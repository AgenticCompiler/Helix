# CLI Optimize Refactor Layering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move optimize-related orchestration out of `src/triton_agent/cli.py` into focused command and domain modules while preserving current CLI behavior.

**Architecture:** Keep `cli.py` as the parser and top-level dispatcher, add a dedicated optimize command handler, and extract optimize runtime, batch, status, rendering, and validation into a focused `optimize/` package. Preserve current external behavior and shift tests toward the extracted modules where they become easier to reason about.

**Tech Stack:** Python, `argparse`, `unittest`, existing Triton agent command modules

---

### Task 1: Add characterization tests for the extracted optimize boundaries

**Files:**
- Modify: `tests/test_cli.py`
- Create: `tests/test_optimize_batch.py`
- Create: `tests/test_optimize_status.py`

- [ ] **Step 1: Write failing tests for optimize-specific helpers moving out of `cli.py`**

Add tests that directly cover:
- batch operator file auto-detection and generated-artifact exclusion
- batch failure summary extraction
- optimize status round parsing and best-round selection

- [ ] **Step 2: Run the focused tests to verify they fail for the missing modules**

Run: `uv run python -m unittest tests.test_optimize_batch tests.test_optimize_status -v`
Expected: FAIL with import errors or missing symbols for the new modules

- [ ] **Step 3: Keep existing CLI behavior tests in place as high-level regression coverage**

Do not remove optimize CLI tests from `tests/test_cli.py`; use them as integration checks while extracting logic.

### Task 2: Extract optimize domain models, validation, rendering, and status helpers

**Files:**
- Create: `src/triton_agent/optimize/__init__.py`
- Create: `src/triton_agent/optimize/models.py`
- Create: `src/triton_agent/optimize/validation.py`
- Create: `src/triton_agent/optimize/render.py`
- Create: `src/triton_agent/optimize/status.py`
- Modify: `src/triton_agent/cli.py`
- Test: `tests/test_optimize_status.py`

- [ ] **Step 1: Create optimize-only dataclasses in `src/triton_agent/optimize/models.py`**

Move:
- `BatchOptimizeWorkspace`
- `BatchOptimizeResult`
- `OptimizeStatusRound`
- `OptimizeStatusWorkspace`

- [ ] **Step 2: Move optimize argument validation into `src/triton_agent/optimize/validation.py`**

Expose a function matching the current CLI behavior for:
- `--min-rounds`
- `--max-concurrency`
- `--continue` incompatibility with explicit mode overrides

- [ ] **Step 3: Move optimize status analysis helpers into `src/triton_agent/optimize/status.py`**

Expose focused functions for:
- inspecting one workspace
- selecting perf artifacts
- parsing `opt-note.md`
- computing mean values and best rounds

- [ ] **Step 4: Move optimize output helpers into `src/triton_agent/optimize/render.py`**

Expose rendering functions for:
- batch optimize results
- optimize status reports

- [ ] **Step 5: Run focused tests to verify the extracted status helpers pass**

Run: `uv run python -m unittest tests.test_optimize_status -v`
Expected: PASS

### Task 3: Extract optimize runtime and batch orchestration

**Files:**
- Create: `src/triton_agent/optimize/orchestration.py`
- Create: `src/triton_agent/optimize/batch.py`
- Modify: `src/triton_agent/cli.py`
- Test: `tests/test_optimize_batch.py`

- [ ] **Step 1: Write the minimal optimize runtime module**

Move the single-workspace optimize lifecycle:
- optimize request building
- skill staging
- temporary optimize `AGENTS.md` lifecycle
- `OptimizeRunLoop` invocation

- [ ] **Step 2: Write the batch orchestration module**

Move:
- workspace scan
- operator candidate resolution
- prefixed streaming helper
- result aggregation and failure summarization

- [ ] **Step 3: Run the focused batch tests**

Run: `uv run python -m unittest tests.test_optimize_batch -v`
Expected: PASS

### Task 4: Add optimize command handler and thin the CLI entrypoint

**Files:**
- Create: `src/triton_agent/commands/__init__.py`
- Create: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add an optimize command handler module**

Expose functions for:
- `handle_optimize`
- `handle_optimize_batch`
- `handle_optimize_status`

- [ ] **Step 2: Update `src/triton_agent/cli.py` to dispatch optimize commands through the new handler**

Keep `cli.py` responsible for parser construction and top-level command routing only.

- [ ] **Step 3: Run the CLI regression tests for optimize commands**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

### Task 5: Run repository verification and review remaining CLI boundaries

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `README.md` if needed
- Modify: `AGENTS.md` if needed

- [ ] **Step 1: Re-read the design doc and confirm the implementation stayed inside the intended optimize-only scope**

Check: `docs/notes/2026-04-07-cli-optimize-refactor-layering.md`

- [ ] **Step 2: Run repository verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 3: If verification reveals extraction regressions, fix them without broadening the refactor beyond optimize-related concerns**
