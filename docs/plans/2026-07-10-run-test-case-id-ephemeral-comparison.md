# Run-Test Case-Id Ephemeral Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run-test --case-id` reuse matching reference cases when possible and otherwise compare a single rerun case without persisting `*_result.pt` artifacts.

**Architecture:** Keep the top-level CLI thin by adding skill-side helpers for single-case payload execution and payload-level comparison, then thread those helpers through both the staged run-eval CLI and the top-level `run-test` command. Preserve the existing file-based archive flow for runs without `--case-id`.

**Tech Stack:** Python, argparse, unittest/pytest-style tests, staged run-eval skill scripts.

---

### Task 1: Lock the new single-case contract in tests

**Files:**
- Modify: `tests/test_execution_commands.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_test_runner.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write failing tests for single-case ref reuse and no-artifact mode**
- [ ] **Step 2: Run the focused tests and verify they fail for the expected missing behavior**

### Task 2: Add skill-side single-case payload helpers

**Files:**
- Modify: `skills/common/ascend-npu-run-eval/scripts/test_runner.py`
- Modify: `skills/common/ascend-npu-run-eval/scripts/compare_result.py`
- Modify: `skills/common/ascend-npu-run-eval/scripts/npu_compare.py` if a public payload-selection helper is needed

- [ ] **Step 1: Add single-case payload execution helpers without `*_result.pt` persistence**
- [ ] **Step 2: Add helpers to extract one case from an existing payload and compare payload objects directly**
- [ ] **Step 3: Run focused skill-script tests and make them pass**

### Task 3: Thread single-case ephemeral comparison through both CLIs

**Files:**
- Modify: `skills/common/ascend-npu-run-eval/scripts/cli.py`
- Modify: `src/helix/eval/runners.py`
- Modify: `src/helix/commands/comparison.py`
- Modify: `src/helix/commands/execution.py`

- [ ] **Step 1: Route `--case-id` through the new ephemeral comparison path**
- [ ] **Step 2: Reuse existing ref cases, rerun missing ref cases only when `--ref-operator-file` is available, and suppress archived-result output in single-case mode**
- [ ] **Step 3: Run focused top-level and staged CLI tests and make them pass**

### Task 4: Update docs and final verification

**Files:**
- Modify: `README.md`
- Modify: `skills/common/ascend-npu-run-eval/references/run-test.md`
- Modify: `docs/specs/2026-07-09-run-test-case-id-and-verbose-guidance-design.md`

- [ ] **Step 1: Document the new single-case no-artifact and ref-reuse semantics**
- [ ] **Step 2: Run targeted verification, changed skill-script strict pyright, and `uv run --group dev ruff check`**
- [ ] **Step 3: Record any environment-blocked full-suite checks explicitly**
