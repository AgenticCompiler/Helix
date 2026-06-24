---
name: triton-npu-repair-guide
description: >-
  Heuristic fixes for Ascend Triton compile/JIT/kernel errors and numerical precision mismatches when editing or converting the operator.
  Does not replace npu-gen-test, npu-gen-bench, or harness specs.
---

# Triton repair experience (Ascend)

Use this skill when **repairing the operator** after Triton/Ascend **compilation**,
**JIT**, **kernel-side** failures, or **numerical mismatches** vs the torch baseline
— especially during `npu-gen-eval-suite`, `convert`, `optimize`, or any flow
that exercises the real Triton path.

Patterns and code hints live in [references/repair-experience.md](references/repair-experience.md).
Match the error or symptom, apply a **minimal** change, then re-run validation.

## Relationship to other skills

- Does **not** replace `npu-gen-test`, `npu-gen-bench`, or normative harness specs.
- Complements `npu-run-eval` (re-validate after applying a heuristic).
- When a generation-only workflow such as `npu-gen-test` or `npu-gen-bench` stages this skill, use it as a diagnostic reference for compile, JIT, launch, kernel-side, or numerical symptoms.

## How to apply

1. Open [references/repair-experience.md](references/repair-experience.md) and match error text or symptom to a section.
2. Apply the smallest change; re-run validation through the `npu-run-eval` skill, using `run-test-baseline` or `run-test-optimize` / `run-bench` as appropriate.
3. If nothing fits, do **not** force a heuristic—fall back to logs, IR skills, or deeper debugging.

## Append-Only Repair Log

If you later **fix** the operator successfully with a **new** pattern not covered above, append a short entry to [output.md](output.md).

Start each new block with:

```text
----- <short title> ----
```

Then add a few lines covering the symptom, the fix, and how you verified it. **Append only**—do not delete or rewrite older blocks.
