---
name: triton-npu-prepare-optimize-baseline
description: Establish a reusable canonical optimize baseline by reusing or generating harnesses, performing minimum repair, and passing `check-baseline`.
---

# Prepare Optimize Baseline

## Goal

Establish a reusable canonical `baseline/` before any optimize round begins.

Use this skill when optimize work cannot yet start because baseline artifacts are missing, invalid, or no longer match the current operator workspace.

## Outputs

- reusable correctness and benchmark harnesses
- `baseline/`
- `baseline/state.json`
- `baseline/perf.txt`

## Workflow

### 1. Inspect And Reuse

- Inspect the operator workspace before generating anything new.
- Reuse existing correctness and benchmark harnesses when they already validate the current operator workspace.
- If a usable correctness harness is missing, use the sibling `triton-npu-gen-test` skill.
- If a usable benchmark harness is missing, use the sibling `triton-npu-gen-bench` skill.

### 2. Reach A Benchmarkable Start

- Use the sibling `triton-npu-run-eval` skill for correctness validation and benchmark validation.
- If the current operator or harnesses need repair before they validate cleanly, do only the minimum repair needed to reach a correct, benchmarkable starting point.
- Treat this phase as baseline repair, not as an optimization round.

### 3. Write Canonical Baseline Artifacts

- Read `../triton-npu-optimize/references/artifacts.md` before writing `baseline/state.json`.
- Create `baseline/`.
- Write `baseline/state.json`.
- Write `baseline/perf.txt`.
- Keep the canonical baseline artifacts anchored to the operator state that just passed correctness and benchmark validation.

### 4. Gate The Baseline

- Use the sibling `triton-npu-optimize-check` skill to run `check-baseline`.
- Keep repairing baseline state until `check-baseline` passes.
- Stop once the workspace has a reusable canonical baseline.

## Completion Condition

This skill is complete only when:

- the workspace has reusable correctness and benchmark harnesses
- `baseline/` exists
- `baseline/state.json` exists and matches the optimize artifact contract
- `baseline/perf.txt` exists
- `triton-npu-optimize-check` `check-baseline` passes

## Hard Rules

- Do not start `opt-round-N/` from this skill.
- Do not do open-ended optimization work here.
- Do not skip benchmark validation.
- Do not treat a partially repaired workspace as a reusable baseline.
