---
name: ascend-npu-distill-patterns
description: Distill completed Ascend NPU optimization evidence into reusable optimize-knowledge pattern cards, then support simulation and mismatch analysis loops that verify whether staged skills can reproduce the optimization without seeing the answer.
---

# Ascend NPU Distill Patterns

## Goal

Convert optimization evidence into durable, generic pattern-card guidance for
`<Language>-npu-optimize-knowledge`. Use this skill when a command asks you to
compare a baseline operator with an optimized answer, read a completed optimize
workspace, update pattern cards, simulate applying those cards, or analyze why a
simulated candidate did not match the expected optimization.

## Distillation Workflow

1. Read the baseline and optimized answer first. Identify the actual code-shape
   changes, correctness constraints, and performance intent before editing any
   skill file.
2. For optimize-process inputs, read `opt-note.md` before round files. Use
   `learned_lessons.md`, `opt-round-*/summary.md`, `opt-round-*/attempts.md`,
   `round-state.json`, and optional `perf-analysis.md` as evidence hints, not as
   substitutes for reading the code diff.
3. Match optimization mechanisms semantically against the editable knowledge
   skill's `references/pattern_index.md` and candidate cards under
   `references/patterns/`. Do not rely on filename citations or keyword matches
   alone.
4. If an existing card honestly covers the mechanism, update that card only when
   the evidence exposes missing preconditions, anti-signals, repair steps, or
   verification guidance.
5. If no existing card's `## Summary` and `## Use When` fit, add a new generic
   pattern card in the same knowledge tree. The new card must be reusable beyond
   the current operator.
6. Regenerate or update the pattern index after adding or editing cards when the
   knowledge skill provides an index builder.

## Pattern Card Contract

- Every card starts with `# <Human Title>`.
- The first section is `## Summary`.
- The second section is `## Use When`.
- Keep `## Summary` and `## Use When` concise and retrieval-oriented: include
  applicability conditions, symptoms, and key code-shape cues, but keep detailed
  reasoning, examples, and edge cases in later sections.
- Put detection rules under `## Use When` or `## Signals`.
- Put stop conditions under `## Avoid When`, `## Failure Modes`, or another
  clearly named anti-signal section.
- Put validation rules under `## What To Verify After Applying`.
- Preserve existing valid guidance unless new evidence clearly supersedes it.
- Keep final prose kernel-agnostic and self-contained. Avoid round IDs, artifact
  path narration, and benchmark-specific labels except inside short examples
  where they are necessary to explain the mechanism.

## Simulation Rules

When simulating an optimizer worker, use only the baseline file in the current
directory and the staged skills. Do not read parent directories, optimized answer
files, generated distillation JSON, unified diffs, reports, or other files that
would reveal the answer.

Apply the matched pattern names as retrieval hints, then write the optimized
candidate requested by the command. If a named pattern is not relevant after
reading the baseline, prefer a conservative candidate over forcing an unrelated
transformation.

## Analysis Rules

When auditing a simulated candidate, compare it with the expected optimized
answer by mechanism, not by byte-for-byte equality. Mark it aligned when it
captures the same important code changes and constraints, even if formatting or
irrelevant local structure differs.

If the candidate misses the mechanism, update the editable knowledge skill with
the missing reusable guidance. Explain the gap in terms of semantic
preconditions, code-shape change, observed mismatch, and verification checks for
the next iteration.

If the candidate overgeneralized a pattern, refine the existing card with
`Avoid When`, anti-signals, or verification rules instead of only adding a new
positive example.

## Boundaries

- This skill owns workflow guidance. The CLI owns workspace discovery, staging,
  iteration, reports, and promotion.
- Do not modify the live optimize workflow skill unless the command explicitly
  asks for it.
- Do not create progress tables or benchmark-specific inventories as part of the
  default distillation flow.
