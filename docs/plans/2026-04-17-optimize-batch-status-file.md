# Optimize Batch Status File Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a root-level `optimize-batch-status.json` file so rerunning `optimize-batch` can skip workspaces explicitly marked as completed.

**Architecture:** Keep the feature in batch orchestration. `optimize-batch` reads one root-level status file before scheduling work, skips matching completed workspaces, and updates the file after each workspace result. Rendering changes stay limited to batch optimize output.

**Tech Stack:** Python, `unittest`, existing optimize batch orchestration and rendering modules.

---

### Task 1: Document The Behavior

**Files:**
- Create: `docs/specs/2026-04-17-optimize-batch-status-file-design.md`
- Create: `docs/plans/2026-04-17-optimize-batch-status-file.md`

- [ ] **Step 1: Write the design doc**

Capture the status file path, JSON shape, skip rules, write rules, reset behavior, and output changes.

- [ ] **Step 2: Write the implementation plan**

Describe the code areas, tests, and verification commands needed for this change.

### Task 2: Add Failing Tests For Batch Status Semantics

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write runtime tests for skip and persistence**

Add tests that:
- skip a workspace when `optimize-batch-status.json` marks it completed with the matching operator file
- write `completed` after a successful workspace run
- write `incomplete` after a failed workspace run
- ignore malformed JSON and still run the workspace

- [ ] **Step 2: Write CLI-facing summary tests**

Add tests that assert skipped workspaces render as `SKIP` and that the summary includes the skipped count.

- [ ] **Step 3: Run the targeted tests and watch them fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime tests.test_cli -v
```

Expected:

- New tests fail because batch status file support and skip rendering do not exist yet.

### Task 3: Implement Batch Status File Support

**Files:**
- Modify: `src/triton_agent/optimize/batch.py`
- Modify: `src/triton_agent/optimize/render.py`
- Modify: `src/triton_agent/optimize/models.py`

- [ ] **Step 1: Add batch result state for skipped workspaces**

Extend batch optimize result data so rendering can distinguish success, failure, and skip cleanly.

- [ ] **Step 2: Add status-file load and validation helpers**

Implement helpers that:
- locate `<batch-root>/optimize-batch-status.json`
- parse JSON defensively
- validate `version`
- look up workspace entries by relative path
- require `status == "completed"` and matching `operator_file` for skipping

- [ ] **Step 3: Add reset handling**

When `options.reset_optimize` is true, remove the batch status file before scheduling any workspace work.

- [ ] **Step 4: Add skip scheduling behavior**

Before building an optimize request, decide whether the workspace should be skipped and emit a skipped batch result instead of launching the agent.

- [ ] **Step 5: Add status-file writes after completion**

After each workspace finishes:
- write `completed` on success
- write `incomplete` on failure

- [ ] **Step 6: Update batch rendering**

Render `[SKIP]` lines and `Summary: X succeeded, Y failed, Z skipped`.

- [ ] **Step 7: Run the targeted tests and make them pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime tests.test_cli -v
```

Expected:

- All new batch status tests pass.

### Task 4: Update User-Facing Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the status file**

Explain that `optimize-batch` records explicit completion state in `optimize-batch-status.json` and skips completed workspaces on rerun.

- [ ] **Step 2: Document reset behavior**

Explain that `--reset-optimize` clears both per-workspace optimize artifacts and the batch status file in batch mode.

### Task 5: Verify

**Files:**
- Modify: none

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime tests.test_cli -v
```

- [ ] **Step 2: Run repo verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```

- [ ] **Step 3: Report any gaps**

If a broader verification command cannot run or fails for unrelated reasons, report that clearly in the final handoff.
