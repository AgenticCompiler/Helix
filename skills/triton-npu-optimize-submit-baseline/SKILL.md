---
name: triton-npu-optimize-submit-baseline
description: Submit and validate optimize baseline artifacts with a script-backed contract before optimize rounds may begin or continue.
---

# Optimize Submit Baseline

Submit the current optimize baseline for validation without doing open-ended optimization work.

Use this skill when you need to confirm that `baseline/` is complete, reusable, and acceptable for later optimize rounds.

## Required Script

Use the bundled helper script:

```bash
python3 scripts/optimize_submit_baseline.py check-baseline --baseline-dir baseline
```

## Behavior

- `check-baseline` verifies canonical baseline artifacts and baseline state.
- The CLI prints JSON only; read the `guideline` field for the human-facing pass/fix instruction.
- When the check fails, treat the returned issues as the baseline repair checklist.
- Do not move on to `opt-round-1` or any later round until the baseline submission passes.
- Do not use this skill to invent missing evidence or to replace benchmark, correctness, profile, or IR work that the workflow still requires.
- Do not use this skill to generate missing harnesses, repair operator logic, or invent missing baseline evidence.
- Baseline preparation belongs to `triton-npu-prepare-optimize-baseline`.
- Open-ended optimization work belongs to `triton-npu-optimize`.
