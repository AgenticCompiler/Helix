---
name: triton-npu-kernel-bench-logs
description: Interprets NPUKernelBench-style operator workspaces with baseline, opt-round-N, PyTorch and Triton sources, and perf text logs. Use when reviewing archived optimization runs under trees such as workspace/NPUKernelBench_level_1_2_triton, comparing rounds without rerunning benchmarks, extracting PyTorch timings from raw-op-statistic-case lines, or onboarding from opt-note.md and learned_lessons.md.
---

# Kernel bench logs

## Goal

Read and summarize completed optimization evidence in an NPUKernelBench-style operator directory (for example `workspace/NPUKernelBench_level_1_2_triton/NN_OperatorName/`) without treating the tree as a live optimization workspace unless the user asks to continue rounds.

## When to use

- The user points at `NPUKernelBench_level_1_2_triton` or a sibling bench export and wants an overview, diff narrative, or perf story.
- You need to compare PyTorch reference numbers to initial Triton and final Triton using existing `*_perf.txt` files only.
- You need the fastest on-ramp to what happened across rounds (`opt-note.md`, `learned_lessons.md`, then selective `opt-round-*`).

## Directory map

Follow the canonical path and filename semantics in [references/kernel-bench-layout.md](references/kernel-bench-layout.md). That file lists operator-level paths, optional round artifacts, and how to read PyTorch timing lines.

## Reading order

1. `opt-note.md` for the cross-round ledger and overall outcome.
2. `learned_lessons.md` for distilled, reusable rules (not round play-by-play).
3. `baseline/` versus top-level `triton_NN_OperatorName.py` and `triton_NN_OperatorName_perf.txt` to confirm the starting Triton snapshot and its benchmark.
4. Top-level `opt_triton_NN_OperatorName.py` and `opt_triton_NN_OperatorName_perf.txt` for the latest optimized snapshot as archived in that tree.
5. `opt-round-{i}/` in ascending `i` when the user needs attempt-level detail: `attempts.md`, `summary.md`, `round-state.json`, then optional deeper artifacts (`perf-analysis.md`, `profile/`, `ir/`) when present.
6. `NN_OperatorName.json` when shapes, dtypes, or case indices need to be tied back to perf lines.
7. `NN_OperatorName.py` and `NN_OperatorName_perf.txt` for the PyTorch reference implementation and its benchmark export.

## Perf text conventions

- Triton and optimized Triton benchmark exports use the same `*_perf.txt` pattern family as the PyTorch file: locate per-case blocks and any `raw-op-statistic-case-*` comment lines when you need structured timing JSON.
- For PyTorch timing specifically, read `avg_time_us` (and related fields) under each `raw-op-statistic-case-*` comment’s `"ops"` array in `NN_OperatorName_perf.txt`, as described in the layout reference.
- `opt-round-{i}/logs/compare-perf.txt` may be absent. When it exists, treat it as a convenient comparison transcript; when it does not, rely on `summary.md`, `attempts.md`, `round-state.json`, paired `*_perf.txt` exports, and `opt-note.md` instead of fabricating a consolidated speedup table.

## Boundaries

- Ignore `optimize-logs/` unless the user explicitly requests it.
- When `logs/compare-perf.txt` exists for a round, cite it instead of rebuilding the same comparison by hand from raw perf exports.
- For live optimization work, follow `triton-npu-optimize` and its artifact contract; use this skill for retrospective log reading and cross-run synthesis.

## Related skills

- `triton-npu-optimize` for executing new optimization rounds and live artifact discipline.
