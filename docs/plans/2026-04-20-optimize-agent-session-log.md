# Optimize Agent Session Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record each optimize code agent launch with timestamp, role, session id, and agent backend.

**Architecture:** Reuse the existing optimize archive run directory and append compact JSONL entries after every optimize agent invocation. Keep Codex session extraction backend-local and keep log writing in optimize execution/guidance code.

**Tech Stack:** Python `unittest`, `pathlib`, `json`, existing optimize guidance and execution modules

---

### Task 1: Parse Codex Startup Session Id

**Files:**
- Modify: `tests/test_codex_runner.py`
- Modify: `src/helix/backends/codex.py`

- [ ] Add a failing test for `session id: <uuid>` startup text.
- [ ] Run `uv run python -m unittest tests.test_codex_runner -v` and verify the test fails.
- [ ] Extend Codex session id extraction to parse UUID tokens from that line.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Add Session Log Paths And Writer

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `src/helix/optimize/guidance.py`

- [ ] Add failing tests for the session log path and JSONL writer.
- [ ] Run `uv run python -m unittest tests.test_optimize_guidance -v` and verify failure.
- [ ] Add `agent_sessions_path` to supervised guidance state and a writer helper for `timestamp`, `role`, `session_id`, `agent`.
- [ ] Re-run the focused test and verify it passes.

### Task 3: Record Optimize Agent Launches

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/helix/optimize/execution.py`

- [ ] Add failing supervised tests that record worker and supervisor launches.
- [ ] Add failing unsupervised tests that record worker launch with `unknown` fallback.
- [ ] Run `uv run python -m unittest tests.test_optimize_runtime -v` and verify failure.
- [ ] Append a JSONL record after each optimize `run()` or `resume()` returns.
- [ ] Re-run the focused runtime tests and verify they pass.

### Task 4: Verify Focused Suite

**Files:**
- No code changes expected.

- [ ] Run `uv run python -m unittest tests.test_codex_runner tests.test_optimize_guidance tests.test_optimize_runtime -v`.
- [ ] Fix any regressions.
