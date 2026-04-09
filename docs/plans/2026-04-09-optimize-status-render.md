# Optimize Status Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder `optimize-status` output to show `NO-SESSION` first and add TTY-only color styling that distinguishes titles, content, and warning detail.

**Architecture:** Keep the change isolated to the optimize render layer. Add a deterministic state-priority sort key plus TTY-aware ANSI styling helpers in `src/triton_agent/optimize/render.py`, then cover them with focused render tests and a light CLI regression check.

**Tech Stack:** Python 3.11, `unittest`, `io.StringIO`, existing optimize render helpers

---

### Task 1: Lock Render Ordering In Tests

**Files:**
- Create: `tests/test_optimize_render.py`
- Modify: `src/triton_agent/optimize/render.py`

- [ ] **Step 1: Write the failing test**

Add a render test that passes three workspaces in mixed order and asserts the rendered order is:
- `NO-SESSION`
- `WARN`
- `OK`

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_render.OptimizeRenderTests.test_render_optimize_status_groups_no_session_then_warning_then_ok -v`
Expected: FAIL because the renderer still sorts purely by workspace name

- [ ] **Step 3: Write minimal implementation**

Add a small state-priority sort key in the render layer.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_render.OptimizeRenderTests.test_render_optimize_status_groups_no_session_then_warning_then_ok -v`
Expected: PASS

### Task 2: Lock TTY Color Behavior In Tests

**Files:**
- Create: `tests/test_optimize_render.py`
- Modify: `src/triton_agent/optimize/render.py`

- [ ] **Step 1: Write the failing tests**

Add tests that require:
- non-TTY render output to contain no ANSI escape codes
- TTY render output to colorize title lines
- TTY warning lines to use a faint gray style

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_render.OptimizeRenderTests.test_render_optimize_status_uses_tty_colors_and_plain_redirect_output -v`
Expected: FAIL because the renderer currently emits plain text in all modes

- [ ] **Step 3: Write minimal implementation**

Add TTY-aware styling helpers and wire them into title, detail, warning, and summary lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_render -v`
Expected: PASS

### Task 3: Verify CLI Integration

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add or update one CLI regression assertion**

Keep the plain-text integration path covered by asserting `NO-SESSION` still appears and sorting stays sensible without color assumptions.

- [ ] **Step 2: Run focused verification**

Run: `uv run python -m unittest tests.test_optimize_render tests.test_cli.PathResolutionTests.test_main_optimize_status_reports_no_session_workspaces -v`
Expected: PASS

- [ ] **Step 3: Run repo verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
