# Code Agent Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one shared retry mechanism for CLI-launched code agents, controlled by an environment variable, and remove optimize-specific 429 retry duplication.

**Architecture:** Keep retry policy at the `AgentRunner` boundary so all CLI backends share it while `OpenHandsRunner` stays unchanged. Narrow `OptimizeRunLoop` to orchestration-owned recovery only: stalled runs, round continuation, and supervisor decisions.

**Tech Stack:** Python `unittest`, existing backend runner abstractions, Python `os.environ`, Python `time.sleep`

---

### Task 1: Add shared backend retry coverage

**Files:**
- Modify: `tests/test_backends_base.py`

- [ ] **Step 1: Write failing tests for shared retry behavior**
- [ ] **Step 2: Run `uv run python -m unittest tests.test_backends_base -v` and verify failure**
- [ ] **Step 3: Cover default retries, env override, `0` disabling retry, interactive bypass, and invalid env values**

### Task 2: Implement shared retry in the backend base layer

**Files:**
- Modify: `src/triton_agent/backends/base.py`

- [ ] **Step 1: Add env-var parsing and transient-failure detection helpers**
- [ ] **Step 2: Wrap non-interactive `run()` execution in a shared retry loop with exponential backoff**
- [ ] **Step 3: Keep verbose launch logging single-shot and preserve final `AgentResult` semantics**

### Task 3: Remove optimize-owned 429 retry logic

**Files:**
- Modify: `src/triton_agent/optimize/run_loop.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing tests that reflect stall-only optimize recovery semantics**
- [ ] **Step 2: Remove rate-limit/backoff logic from `OptimizeRunLoop`**
- [ ] **Step 3: Keep stall recovery, min-round continuation, and supervisor decision handling intact**

### Task 4: Verify the change end to end

**Files:**
- Modify: `tests/test_backends_base.py`
- Modify: `tests/test_supervisor.py`
- Modify: `src/triton_agent/backends/base.py`
- Modify: `src/triton_agent/optimize/run_loop.py`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_backends_base tests.test_supervisor -v`**
- [ ] **Step 2: Inspect output and fix any regressions**
- [ ] **Step 3: Report final behavior, including the new `TRITON_AGENT_CODE_AGENT_MAX_RETRIES` semantics**
