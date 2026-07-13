# Optimize Round Timing JSONL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-round optimize timing as JSONL files under `.helix/round-timings/`, stop storing round timestamps in workflow state, and archive the timing directory directly.

**Architecture:** Keep workflow state as the source of truth for active-round coordination while moving historical timing into per-round JSONL files. Write round lifecycle events from the optimize-state skill, write validation command events from the run-eval skill, and copy the full timing directory during optimize archive cleanup.

**Tech Stack:** Python, `unittest`, optimize-state skill scripts, run-eval skill scripts

---

### Task 1: Lock Down Docs

**Files:**
- Create: `docs/specs/2026-07-06-optimize-round-timing-jsonl-design.md`
- Create: `docs/plans/2026-07-06-optimize-round-timing-jsonl.md`

- [ ] **Step 1: Write the design doc**

Capture the per-round JSONL layout, event types, state contract change, legacy compatibility rule, and archive-directory behavior.

- [ ] **Step 2: Write the implementation plan**

Record the test-first file list, focused code changes, and verification commands.

### Task 2: Add Failing Regression Coverage

**Files:**
- Modify: `tests/test_optimize_workflow_state.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Add workflow-state timing-log regressions**

Add tests that `start_round()` creates `.helix/round-timings/opt-round-N.jsonl`, appends a `round_start` event, and writes round state without `started_at` or `ended_at`. Add a matching completion test for `round_end`, plus a legacy-state load test that still accepts old timestamp fields.

- [ ] **Step 2: Add run-eval timing-log regressions**

Add script-level tests that `run-test-optimize` and `run-bench` append start and end events for the currently active round while preserving command exit behavior.

- [ ] **Step 3: Add archive regression**

Replace the existing archive assertion that expects `round-timings.json` with one that expects `helix-logs/<run_id>/round-timings/opt-round-N.jsonl`.

- [ ] **Step 4: Run focused tests to confirm failure first**

Run:
`uv run python -m unittest tests.test_optimize_workflow_state tests.test_skill_command_script tests.test_optimize_guidance -v`

Expected: FAIL on the new timing-log assertions before implementation.

### Task 3: Implement Per-Round Timing Logs

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`
- Modify: `skills/common/ascend-npu-run-eval/scripts/cli.py`
- Modify: `src/helix/optimize/archive.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/hook_runtime/optimize/workflow_state.py`
- Modify: `src/helix/optimize/workflow_state.py`

- [ ] **Step 1: Write round lifecycle JSONL events from optimize-state**

Add helpers in `state_machine.py` to resolve `.helix/round-timings/opt-round-N.jsonl`, append compact JSONL events, emit `round_start` from `start_round()`, emit `round_end` from `complete_round()`, and stop writing `started_at` / `ended_at` into new workflow-state payloads.

- [ ] **Step 2: Keep workflow-state loading backward compatible**

Relax validation so legacy round entries may still contain `started_at` and `ended_at`, but new state writes no longer require or depend on them.

- [ ] **Step 3: Write run-test and run-bench JSONL events from run-eval**

Add best-effort helpers in `skills/common/ascend-npu-run-eval/scripts/cli.py` that discover an active optimize round from `.helix/state.json`, then append `run_test_start` / `run_test_end` and `run_bench_start` / `run_bench_end` around the existing command execution flow.

- [ ] **Step 4: Archive the timing directory directly**

Update optimize archive/session cleanup code to copy `.helix/round-timings/` into `helix-logs/<run_id>/round-timings/` and remove the old single-file `round-timings.json` archive behavior.

- [ ] **Step 5: Re-run the focused tests**

Run:
`uv run python -m unittest tests.test_optimize_workflow_state tests.test_skill_command_script tests.test_optimize_guidance -v`

Expected: PASS

- [ ] **Step 6: Run the required strict Pyright checks for touched skill scripts**

Run:
`bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/cli.py`

### Task 4: Run Repository Verification

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`
- Modify: `skills/common/ascend-npu-run-eval/scripts/cli.py`
- Modify: `src/helix/optimize/archive.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/hook_runtime/optimize/workflow_state.py`
- Modify: `src/helix/optimize/workflow_state.py`
- Modify: `tests/test_optimize_workflow_state.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Run Ruff**

Run:
`uv run --group dev ruff check`

Expected: PASS

- [ ] **Step 2: Run Pyright**

Run:
`uv run pyright`

Expected: PASS

- [ ] **Step 3: Run pytest**

Run:
`uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: PASS
