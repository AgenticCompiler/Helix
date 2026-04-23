---
name: triton-npu-analyze-compiler-source
description: Use when an optimize round has CLI-provisioned AscendNPU-IR compiler source and needs source-backed explanation for a compiler error, suspicious IR pass transition, lowering symptom, or stalled optimization direction that profile and IR evidence alone cannot explain.
---

# Analyze Compiler Source

Use this skill to explain one concrete Triton Ascend optimization symptom with evidence from the CLI-provisioned AscendNPU-IR checkout.

AscendNPU-IR is the compiler-side source tree used to understand MLIR-style lowering, optimization passes, legality checks, layout transforms, and target-specific mapping behavior. Treat it as evidence for explaining compiler behavior, not as a place to patch code during operator optimization.

## Inputs

- The current operator workspace and `opt-round-N/` directory.
- A concrete compiler error, IR stage transition, pass name, op name, lowering symptom, or stalled optimization hypothesis.
- The CLI-provided local source path and commit from the launch prompt or workspace guidance.
- Existing benchmark, profiler, and IR artifacts when available.

## When To Use

Use this skill only when compiler source analysis is enabled by the launch prompt or workspace guidance and at least one condition applies:

- A compiler, lowering, or legality failure blocks the round and the error text alone does not explain the fix.
- IR evidence shows suspicious layout conversion, copy insertion, synchronization, buffer expansion, vectorization loss, fusion loss, or pass-to-pass behavior.
- Profiling plus IR identifies a symptom but not why the compiler produced that lowering.
- Multiple attempts have stalled because the current evidence does not identify a concrete next operator change.
- Chip-specific behavior remains unexplained after architecture notes and IR inspection.

## When Not To Use

- Do not use compiler source as the first analysis step.
- Do not use this skill when benchmark, profiler, or IR evidence already supports a concrete next change.
- Do not browse the source tree broadly without a narrowed error, stage, pass, op, or lowering symptom.
- Do not use this skill when the launch prompt or guidance does not say compiler source analysis is enabled.

## Working Rules

- Use only the CLI-provided local source path.
- Treat the compiler source checkout as read-only.
- Do not run `git clone`, `git fetch`, or `git pull`.
- Do not modify files inside the compiler source checkout.
- Do not rely on a repository URL from memory or external instructions.
- Cite local source paths and the compiler source commit for any source-backed claim.
- If the checkout commit cannot be matched to the installed compiler or toolchain version, record the version mismatch as an evidence gap and phrase conclusions as source-informed hypotheses.
- Detailed source indexing is deferred; use any future bundled helper or index only when it is already present.

## Analysis Workflow

1. Confirm the trigger.
2. Record the CLI-provided local source path and commit.
3. Start from round-local evidence:
   - benchmark or correctness behavior
   - profiler artifacts under `opt-round-N/profile/`
   - IR artifacts under `opt-round-N/ir/`
   - compiler error logs
4. Narrow the source search to a stage, pass, op, diagnostic, transform, or subsystem before reading source files.
5. Inspect only the source files needed to explain the narrowed symptom.
6. Separate direct facts from inference.
7. Translate the source-backed explanation into a concrete operator-level implication.
8. Recommend the next operator change or state why the evidence is still insufficient.
9. Write `opt-round-N/compiler-analysis.md`.

## Output Contract

When this skill is used, write a standalone Markdown artifact:

```text
opt-round-N/compiler-analysis.md
```

Use these sections:

1. `# Compiler Source Analysis`
2. `## Executive Summary`
3. `## Trigger`
4. `## Compiler Source Context`
5. `## IR Or Error Evidence`
6. `## Source Files Inspected`
7. `## Source-Backed Explanation`
8. `## Impact On Current Operator`
9. `## Recommended Next Change`
10. `## Confidence And Evidence Gaps`

## Evidence Quality Rules

- Every nontrivial compiler-source conclusion must cite a local source file path and the inspected commit.
- Keep source evidence tied to the current operator and round; do not write general compiler notes unless they affect the next optimization decision.
- Prefer short summaries over long pasted code.
- State when a conclusion depends on inference from source naming, pass structure, or nearby code rather than direct proof.
- Link or summarize `compiler-analysis.md` from `perf-analysis.md`, `attempts.md`, or `summary.md` only when it materially affected the round decision.
