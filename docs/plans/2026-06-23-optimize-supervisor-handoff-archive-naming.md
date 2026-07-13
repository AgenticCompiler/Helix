# Optimize Supervisor Handoff Archive Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename optimize supervisor snapshot archives to `supervisor-handoffs/` and stop creating empty archive directories when no handoff files exist.

**Architecture:** Keep the change local to optimize session artifact coordination, archive copying, and the tests that describe the archive contract. Treat the rename as a semantic cleanup of supervisor-only handoff snapshots rather than a broader history refactor.

**Tech Stack:** Python 3.11, `pathlib`, `unittest`, optimize runtime/session-artifact modules, repository docs

**Implementation note:** Do not create commits unless the user explicitly asks for them.

---

## File Map

- Modify: `src/helix/optimize/archive.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `docs/specs/2026-06-23-optimize-supervisor-handoff-archive-naming-design.md`

### Task 1: Lock the new archive contract with tests

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] Change the no-handoff supervised cleanup assertions to expect no archive handoff directory.
- [ ] Change the real handoff snapshot assertions to expect `supervisor-handoffs/`.
- [ ] Update helper state construction to use renamed handoff fields and private directory names.

### Task 2: Implement the rename and conditional directory creation

**Files:**
- Modify: `src/helix/optimize/archive.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/execution.py`

- [ ] Rename supervisor snapshot fields, parameters, and hidden runtime paths from `history` to `handoff` where they describe supervisor snapshots.
- [ ] Copy snapshot files into `supervisor-handoffs/` only when at least one file exists.
- [ ] Keep archive overwrite protection and non-handoff files unchanged.

### Task 3: Verify the behavior

**Files:**
- Modify: `docs/specs/2026-06-23-optimize-supervisor-handoff-archive-naming-design.md`

- [ ] Run the targeted optimize guidance/runtime tests that cover archive creation and snapshot copying.
- [ ] If the targeted tests pass, summarize the final archive contract in the design note and report the verification results.
