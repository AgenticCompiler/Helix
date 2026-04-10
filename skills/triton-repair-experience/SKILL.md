---
name: triton-repair-experience
description: >-
  Heuristic fixes for Ascend Triton compile/JIT/kernel errors when editing the operator.
  Does not replace test-gen, bench-gen, or harness specs.
---

# Triton repair experience (Ascend)

Use this skill when **repairing the operator** after Triton/Ascend **compilation**, **JIT**, or **kernel-side** failures—especially during `eval-gen`, `optimize`, or any flow that exercises the real Triton path.

Patterns and code hints live in [references/repair-experience.md](references/repair-experience.md). Match the error or symptom, apply a **minimal** change, then re-run validation.

## Relationship to other skills

- Does **not** replace `test-gen`, `bench-gen`, or normative harness specs.
- Complements `operator-eval` (re-validate after applying a heuristic).

## How to apply

1. Open [references/repair-experience.md](references/repair-experience.md) and match error text or symptom to a section.
2. Apply the smallest change; re-run `run-test` / `run-bench` as appropriate.
3. If nothing fits, do **not** force a heuristic—fall back to logs, IR skills, or deeper debugging.

If you later **fix** the operator successfully with a **new** pattern not covered above, follow the **`self-repair`** skill and append a short entry to [../self-repair/output.md](../self-repair/output.md).
