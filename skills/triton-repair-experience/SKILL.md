---
name: triton-repair-experience
description: >-
  Team-maintained Ascend Triton operator repair experience (heuristic fixes for compile/JIT,
  libdevice vs tl.math, tf32 migration, INF dtype, UB overflow, 0d block_type). Use when
  editing the operator kernel to fix Ascend-specific errors; does not override test-gen,
  bench-gen, or normative specs.
---

# Triton repair experience (Ascend)

Use this skill when **repairing the operator** after Triton/Ascend **compilation**, **JIT**, or **kernel-side** failures—especially during `eval-gen`, `optimize`, or any flow that validates the real Triton path.

## What it contains

- **Overview:** match errors or symptoms to a small set of **heuristic** patterns; apply **minimal** changes; re-run validation.
- **Details:** see [references/repair-experience.md](references/repair-experience.md) for the full table and code patterns.

The sibling skill **`ascend-triton-repair-heuristics`** is a thin pointer to the same reference document; prefer this skill when you want the canonical location for the experience file.

## Relationship to other skills

- Does **not** replace `test-gen`, `bench-gen`, or harness specs.
- Complements `operator-eval` (re-validate after applying a heuristic).

## How to apply

1. Open [references/repair-experience.md](references/repair-experience.md).
2. Match **error text or symptom** to a section.
3. Apply the **smallest** change; re-run `run-test` / `run-bench` as appropriate.
4. If nothing fits, do **not** force a heuristic—use logs, IR skills, or deeper debugging.
