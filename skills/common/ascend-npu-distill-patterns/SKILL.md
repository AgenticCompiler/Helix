---
name: ascend-npu-distill-patterns
description: Distill completed Ascend NPU optimization evidence into reusable optimize-knowledge pattern cards, then support simulation and mismatch analysis loops that verify whether staged skills can reproduce the optimization without seeing the answer.
---

# Ascend NPU Distill Patterns

## Goal

Convert optimization evidence into durable, generic pattern-card guidance for
the active language's `<language>-npu-optimize-knowledge` skill. Use this skill
when a command asks you to compare a baseline operator with an optimized answer,
read a completed optimize workspace, update pattern cards, simulate applying
those cards, or analyze why a simulated candidate did not match the expected
optimization.

## Language Scope

The command prompt names the active operator language and editable knowledge
skill. Use that skill as the source of truth, for example
`triton-npu-optimize-knowledge` for Triton or `tilelang-npu-optimize-knowledge`
for TileLang.

Keep cards reusable within the active language. Do not assume Triton syntax,
`@triton.jit`, or Triton IR when the active language is TileLang. When evidence
shows a cross-language Ascend NPU idea, write the mechanism in language-neutral
terms first, then keep API names, code examples, and verification details tied to
the active language's knowledge skill.

## Distillation Workflow

1. Read the baseline and optimized answer first. Identify the actual code-shape
   changes, correctness constraints, and performance intent before editing any
   skill file.
2. For optimize-process inputs, read `opt-note.md` before round files. Use
   `learned_lessons.md`, `opt-round-*/summary.md`, `opt-round-*/attempts.md`,
   `round-state.json`, and optional `perf-analysis.md` as evidence hints, not as
   substitutes for reading the code diff. For each `opt-round-*/`, compare the
   kernel snapshot to the previous round. Extract only performance-impactful
   mechanisms; drop a change if any filter fails:
   - Round speedup vs the previous round (or baseline for round 1) is below 1.05x.
   - `opt-note.md` or the round summary does not describe it as a meaningful win.
   - The change did not survive to the final optimized code.
   - The change is cosmetic, debug-only, or benchmark showed no gain.
   If no card updates are needed, set `"aligned": true` in the JSON output.
3. Match optimization mechanisms semantically against the editable knowledge
   skill's `references/pattern_index.md` and candidate cards under
   `references/patterns/`. Do not rely on filename citations or keyword matches
   alone. Treat logs and round notes as evidence hints; confirm from the code diff.
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
- Keep final prose operator-agnostic and self-contained within the active
  language. Avoid round IDs, artifact path narration, and benchmark-specific
  labels except inside short examples where they are necessary to explain the
  mechanism.
- Every card must be ≤ 500 lines. Compress in place; never split one card into
  multiple files.
- Do not put operator names, function/variable names, round IDs, benchmark
  numbers, or exact shapes in `## Summary` or `## Use When`.
- Only document kernel performance optimizations — not process, workflow, or
  debugging rules.
- Do not set `priority: high` on new or updated cards.

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
the next iteration. Follow **Pattern Card Contract** when editing cards.

If the candidate overgeneralized a pattern, refine the existing card with
`Avoid When`, anti-signals, or verification rules instead of only adding a new
positive example.

## Post-Update Review

Run when the command asks for a post-update review. Review all updated cards
holistically before export or promotion.

1. **Index first** — read `references/pattern_index.md`; group by domain, flag
   duplicates, non-performance cards, and overly specialized index entries. Do
   not read every full card upfront.
2. **Consolidate selectively** — read full cards only for merge candidates,
   removal candidates, or index red flags. Merge same-mechanism duplicates;
   delete process/methodology cards, overly specialized cards, and near-duplicates.
3. **Fix cross-references** — `## Related Patterns` must be bidirectional; patch
   unchanged cards with missing back-links only; do not heavily rewrite stable cards.
4. **Report** — write JSON describing groups, merges, removals, quality fixes, and
   a brief summary. Edit cards on disk; the JSON describes work already done.

Output shape:

```json
{
  "groups": [{"name": "...", "cards": [], "merged": [], "removed": [], "kept_separate": []}],
  "quality_issues": [{"card": "...", "severity": "error|warning", "category": "...", "detail": "...", "fixed": true}],
  "summary": "..."
}
```

## Boundaries

- This skill owns workflow guidance. The CLI owns workspace discovery, staging,
  iteration, reports, and promotion.
- Do not modify the live optimize workflow skill unless the command explicitly
  asks for it.
- Do not create progress tables or benchmark-specific inventories as part of the
  default distillation flow.
