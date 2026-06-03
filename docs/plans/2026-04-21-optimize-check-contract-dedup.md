# Optimize Check Contract Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate optimize-check models and baseline/round contract parsing while preserving the skill-first validation boundary and direct skill-script execution.

**Architecture:** Move shared contract dataclasses and parsing helpers into a new helper module inside `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/`, then add thin runtime bridge modules in `src/triton_agent/optimize/` that re-export the shared API for existing callers. Keep `optimize_check.py` as a thin CLI wrapper that owns argument parsing and process exit behavior.

**Tech Stack:** Python `dataclasses`, `pathlib`, `importlib`, `unittest`

---

### Task 1: Add failing tests for shared optimize-check contract ownership

**Files:**
- Modify: `tests/test_run_skill_loader.py`
- Modify: `tests/test_optimize_baseline.py`
- Modify: `tests/test_optimize_round_contract.py`

- [ ] Add a failing loader test that asserts the optimize-check skill module and runtime models expose the same `OptimizeCheckResult`, `BaselineState`, and `RoundState` classes.
- [ ] Add a failing baseline test that loads the shared optimize-check helper module and compares its baseline helpers with the runtime wrapper results.
- [ ] Add a failing round test that loads the shared optimize-check helper module and compares its round helpers with the runtime wrapper results.

### Task 2: Move shared contract code into the optimize-check skill

**Files:**
- Create: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_contract.py`
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check.py`

- [ ] Create a shared skill-local helper module containing the optimize-check dataclasses and baseline/round helper logic.
- [ ] Reduce `optimize_check.py` to a thin CLI wrapper that imports shared types and check functions from the helper.
- [ ] Keep direct script execution behavior unchanged.

### Task 3: Replace runtime duplicates with bridge wrappers

**Files:**
- Create: `src/triton_agent/optimize/skill_contract.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/baseline.py`
- Modify: `src/triton_agent/optimize/round_contract.py`

- [ ] Add one runtime bridge module that loads the shared optimize-check helper via the existing skill loader.
- [ ] Re-export shared dataclasses from `models.py`.
- [ ] Re-export shared baseline and round helpers from the existing runtime modules.

### Task 4: Verify focused regressions

**Files:**
- Modify only if verification reveals gaps

- [ ] Run focused unit tests for loader, optimize baseline, round contract, optimize checks, and skill command script coverage.
- [ ] Run lint and type verification for the touched code.
