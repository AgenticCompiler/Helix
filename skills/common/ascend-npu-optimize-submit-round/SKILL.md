---
name: ascend-npu-optimize-submit-round
description: Submit and validate optimize round artifacts with a script-backed contract before the workflow may continue.
---

# Optimize Submit Round

Submit one completed `opt-round-N/` directory for validation without doing open-ended optimization work.

Use this skill when you need to check whether the current round is acceptable before the optimize workflow may continue or stop.

## Required Script

Use the bundled helper script:

```bash
python3 scripts/optimize_submit_round.py check-round --round-dir opt-round-1
python3 scripts/optimize_submit_round.py check-round --round-dir opt-round-1 --optimize-target kernel
python3 scripts/optimize_submit_round.py check-round --round-dir opt-round-2 --current-round 2 --final-round 4
```

## Behavior

- `check-round` verifies round-local artifacts and the recorded round state against the canonical baseline contract.
- When `--current-round` and `--final-round` are provided, `check-round` emits next-step guidance relative to the current worker batch instead of deciding whether the whole optimize session may stop.
- The CLI prints JSON only; read the `guideline` field for the human-facing pass/fix instruction, and read `next_option` when it is present.
- When `--optimize-target kernel` is provided, `check-round` still allows rounds whose recorded `effective_metric_source` fell back to `total-op` or `mixed`, but returns that mismatch as a warning-style issue so the caller can surface it.
- When the check fails, treat the returned issues as the round repair checklist.
- Do not start the next optimize round until this submission passes.
- Do not use this skill to invent missing evidence or to replace benchmark, correctness, profile, or IR work that the workflow still requires.
- Do not use this skill to generate missing harnesses, repair operator logic, or invent missing baseline evidence.
- Baseline preparation belongs to `ascend-npu-prepare-optimize-baseline`.
- Open-ended round analysis belongs to the corresponding `<Language>-npu-optimize` skill (where `<Language>` is `triton` or `tilelang`).
