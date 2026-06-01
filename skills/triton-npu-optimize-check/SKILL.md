---
name: triton-npu-optimize-check
description: Validate optimize baseline and round artifacts with a shared script-backed contract before a worker may continue.
---

# Optimize Check

Validate optimize workflow artifacts without doing open-ended optimization work.

Use this skill from optimize workers and supervisors when you need to check whether:

- `baseline/` is complete and reusable
- one `opt-round-N/` directory is acceptable before the workflow may continue

## Required Script

Use the bundled helper script:

```bash
python3 scripts/optimize_check.py check-baseline --baseline-dir baseline
python3 scripts/optimize_check.py check-round --round-dir opt-round-1
python3 scripts/optimize_check.py check-round --round-dir opt-round-1 --optimize-target kernel
```

## Behavior

- `check-baseline` verifies canonical baseline artifacts and baseline state.
- `check-round` verifies round-local artifacts and the recorded round state against the canonical baseline contract.
- The CLI prints JSON only; read the `guideline` field for the human-facing pass/fix instruction, and read `next_option` when it is present.
- When `--optimize-target kernel` is provided, `check-round` still allows rounds whose recorded `effective_metric_source` fell back to `total-op` or `mixed`, but returns that mismatch as a warning-style issue so the caller can surface it.
- When a check fails, treat the returned issues as the repair checklist.
- Do not start the next optimize round until the current check passes.
- Do not use this skill to invent missing evidence or to replace benchmark, correctness, profile, or IR work that the workflow still requires.
- Do not use this skill to generate missing harnesses, repair operator logic, or invent missing baseline evidence.
- Baseline preparation belongs to `triton-npu-prepare-optimize-baseline`.
- Open-ended round analysis belongs to `triton-npu-optimize`.
