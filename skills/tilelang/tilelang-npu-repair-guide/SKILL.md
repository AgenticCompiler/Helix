---
name: tilelang-npu-repair-guide
description: >-
  Heuristic fixes for TileLang Ascend compile/JIT/kernel errors and numerical precision mismatches when editing or converting the operator.
  Does not replace npu-gen-test, npu-gen-bench, or harness specs.
---

# TileLang Repair Experience (Ascend)

Use this skill when **repairing the operator** after TileLang/Ascend **compilation**,
**JIT**, **kernel-side** failures, or **numerical mismatches** vs the torch baseline
— especially during `npu-gen-eval-suite`, `convert`, `optimize`, or any flow
that exercises the real TileLang path.

## Content (to be populated)

- `references/repair-experience.md` — TileLang-specific compile/JIT/kernel error heuristics.
- `output.md` — append-only log of new fixes discovered.

## Relationship to other skills

- Does **not** replace `npu-gen-test`, `npu-gen-bench`, or normative harness specs.
- Complements `npu-run-eval` (re-validate after applying a heuristic).

## How to apply

1. Open `references/repair-experience.md` and match error text or symptom.
2. Apply the smallest change; re-run validation through `npu-run-eval`.
3. If nothing fits, fall back to logs, IR skills, or deeper debugging.
