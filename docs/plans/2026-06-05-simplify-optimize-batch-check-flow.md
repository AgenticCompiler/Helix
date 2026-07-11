# Simplify Optimize Batch Check Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify optimize batch checking so the runtime uses a single merged batch check result with `pass` or `fail` semantics.

**Architecture:** Update the skill contracts and runtime normalization to expose `status`, then refactor the optimize batch loop to use one `check_batch_round` helper and a simple single-repair policy. Remove the old gate-decision and repair-pending helpers once the new tests pass.

**Tech Stack:** Python, unittest, argparse, uv, pytest

---

### Task 1: Lock the new check contract in tests

**Files:**
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing tests for `status`-based check results**
- [ ] **Step 2: Run targeted optimize tests and confirm failures mention missing `status` or old follow-up behavior**

### Task 2: Update the skill-side optimize check contracts

**Files:**
- Modify: `skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline_contract.py`
- Modify: `skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline.py`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py`

- [ ] **Step 1: Replace `ok` and `decision` with `status` in the skill-side dataclasses and CLI payloads**
- [ ] **Step 2: Keep CLI exit codes derived from `status`**
- [ ] **Step 3: Run the targeted skill command tests**

### Task 3: Refactor runtime batch checking

**Files:**
- Modify: `src/helix/optimize/checks.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/optimize/prompts.py`

- [ ] **Step 1: Normalize runtime check results to `status`**
- [ ] **Step 2: Replace `_determine_batch_followup` with `check_batch_round`**
- [ ] **Step 3: Inject previous-batch issues into the next worker prompt only when needed**
- [ ] **Step 4: Remove obsolete gate/repair helper code**
- [ ] **Step 5: Run targeted runtime tests**

### Task 4: Verify the repository

**Files:**
- Modify only as needed from prior tasks

- [ ] **Step 1: Run `uv run --group dev ruff check`**
- [ ] **Step 2: Run `uv run pyright`**
- [ ] **Step 3: Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`**
