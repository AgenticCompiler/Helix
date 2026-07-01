# Run ID Second-Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change shared run ID generation to second precision by default and add short suffixes only for same-second collisions in one process.

**Architecture:** Keep the change inside the shared OTEL trace helper so all command families inherit the new format automatically. Lock the behavior with focused helper tests instead of broad command-level format assertions.

**Tech Stack:** Python 3.11, `datetime`, `collections.Counter`, `unittest`, shared trace helper tests

**Implementation note:** Do not create commits unless the user explicitly asks for them.

---

## File Map

- Create: `tests/test_otel_trace.py`
- Modify: `src/triton_agent/otel_trace.py`
- Modify: `docs/specs/2026-06-23-run-id-second-precision-design.md`

### Task 1: Add focused failing tests

**Files:**
- Create: `tests/test_otel_trace.py`

- [ ] Add helper tests for a first same-second allocation and a repeated same-second allocation with a numeric suffix fallback.
- [ ] Cover both prefixed and unprefixed run IDs with a patched clock and isolated collision state.

### Task 2: Implement second-precision IDs

**Files:**
- Modify: `src/triton_agent/otel_trace.py`

- [ ] Replace microsecond formatting with second-precision formatting.
- [ ] Add per-process collision tracking keyed by the base run ID string.
- [ ] Return the base value on first allocation and append `-2`, `-3`, and so on for later same-second collisions.

### Task 3: Verify the shared helper behavior

**Files:**
- Modify: `docs/specs/2026-06-23-run-id-second-precision-design.md`

- [ ] Run the new helper tests plus targeted command tests that rely on generated run IDs.
- [ ] Run static/style checks on the touched files and record the results.
