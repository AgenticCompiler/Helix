# Optimize Graceful Interrupt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optimize-specific graceful `Ctrl+C` handling that sends two `SIGINT` signals to the running code agent before force-killing it.

**Architecture:** Keep interrupt ownership in `process_runner.py` so the subprocess layer can translate one user `Ctrl+C` into a predictable shutdown sequence. Thread an optimize-only interrupt policy from the runner entrypoint so `OptimizeRunLoop` receives a normal interrupted result and does not attempt stall recovery.

**Tech Stack:** Python, `subprocess`, POSIX signals, `unittest`

---

### Task 1: Specify and cover interrupt behavior

**Files:**
- Create: `docs/notes/2026-04-07-optimize-graceful-interrupt.md`
- Modify: `tests/test_process_runner.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing process-runner tests**

Add tests that simulate `KeyboardInterrupt` during buffered and streaming optimize runs and assert the process receives two `SIGINT` deliveries before a final force-kill when needed.

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `uv run python -m unittest tests.test_process_runner tests.test_supervisor -v`
Expected: FAIL because the process runner does not yet support optimize interrupt escalation.

### Task 2: Implement interrupt escalation

**Files:**
- Modify: `src/triton_agent/process_runner.py`
- Modify: `src/triton_agent/backends/codex.py`
- Modify: `src/triton_agent/backends/opencode.py`
- Modify: `src/triton_agent/backends/pi.py`
- Modify: `src/triton_agent/backends/claude.py`

- [ ] **Step 1: Add an opt-in interrupt policy to the shared process runner**

Introduce a small configuration object or parameters that let non-interactive runs request graceful interrupt escalation and process-group signaling.

- [ ] **Step 2: Thread optimize-only policy from backend runners**

Make optimize requests opt into the graceful interrupt sequence while leaving other commands on existing behavior.

- [ ] **Step 3: Run targeted tests to verify the new behavior**

Run: `uv run python -m unittest tests.test_process_runner tests.test_supervisor -v`
Expected: PASS

### Task 3: Document the new behavior

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update user-facing docs**

Document that `optimize` intercepts one `Ctrl+C`, forwards two `SIGINT` signals to the code agent with short waits, and force-kills the agent if it still does not stop.

- [ ] **Step 2: Run focused verification**

Run: `uv run python -m unittest tests.test_process_runner tests.test_supervisor tests.test_cli -v`
Expected: PASS

- [ ] **Step 3: Run repository checks**

Run: `uv run --group dev ruff check`
Run: `uv run pyright`
Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
