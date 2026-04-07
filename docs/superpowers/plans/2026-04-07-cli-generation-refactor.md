# CLI Generation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the remaining generation flow for `gen-test` and `gen-bench` out of `src/triton_agent/cli.py` while preserving current CLI behavior.

**Architecture:** Add a generation command handler plus a generation runtime module that owns output resolution, overwrite protection, `AgentRequest` construction, and agent launch orchestration. Leave `cli.py` as the parser and top-level dispatcher.

**Tech Stack:** Python, `argparse`, `unittest`, existing runner and prompt infrastructure

---

### Task 1: Add failing tests for the generation command/runtime boundary

**Files:**
- Create: `tests/test_generation_commands.py`

- [ ] **Step 1: Write focused tests for generation handlers and helpers**

Cover:
- default output path selection
- overwrite protection
- request construction and dispatch through the new handler

- [ ] **Step 2: Run the focused tests to verify they fail because the new modules do not exist yet**

Run: `uv run python -m unittest tests.test_generation_commands -v`
Expected: FAIL with import errors or missing symbols

### Task 2: Extract generation runtime helpers

**Files:**
- Create: `src/triton_agent/generation.py`
- Modify: `src/triton_agent/cli.py`
- Test: `tests/test_generation_commands.py`

- [ ] **Step 1: Move generation output-path and overwrite logic into `src/triton_agent/generation.py`**

- [ ] **Step 2: Move generation request construction and runner invocation into `src/triton_agent/generation.py`**

- [ ] **Step 3: Run focused generation tests**

Run: `uv run python -m unittest tests.test_generation_commands -v`
Expected: PASS

### Task 3: Add generation command handler and thin the CLI entrypoint

**Files:**
- Create: `src/triton_agent/commands/generation.py`
- Modify: `src/triton_agent/commands/__init__.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add `handle_gen_test` and `handle_gen_bench` in `src/triton_agent/commands/generation.py`**

- [ ] **Step 2: Update `src/triton_agent/cli.py` to dispatch generation commands through the new handler**

- [ ] **Step 3: Keep compatibility wrappers only where current tests still need them**

### Task 4: Re-run verification

**Files:**
- Modify: `src/triton_agent/cli.py`

- [ ] **Step 1: Run focused CLI and generation tests**

Run: `uv run python -m unittest tests.test_cli tests.test_generation_commands -v`
Expected: PASS

- [ ] **Step 2: Run full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
