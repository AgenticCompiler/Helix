# Reduce Avoid Transpose Copy Pattern Design

## Summary

Add a new generic optimize pattern card that teaches agents to replace `movedim(...).contiguous()` style non-last-dimension reduction wrappers with direct `[outer, reduce, inner]` strided reduction kernels over the original contiguous layout.

The new card should preserve the user's detailed reduction guidance, but organize it into the repository's authored pattern-card structure so it can participate in generated index triage without hand-editing derived files.

## Goal

Make this optimization direction discoverable during optimize triage when a non-last-dimension reduction is paying a full transpose or layout-copy cost before the real reduction kernel.

## User-Visible Behavior

- A new pattern card is added under `skills/triton-npu-optimize-knowledge/references/patterns/`.
- The card appears in the generated `pattern_index.md` as a normal-priority pattern.
- The generated summary and `Use When` bullets make the copy-elimination opportunity recognizable before the agent opens the full card.
- The authored card keeps the more detailed material needed to implement and validate the rewrite:
  - the `[outer, reduce, inner]` abstraction
  - host-side shape computation without `movedim(...).reshape(...)`
  - one short before/after rewrite example
  - implementation guidance that defines its kernel variables before using them
  - pitfalls around small `inner_size`, UB pressure, and non-contiguous inputs

## Design

### Card Structure

Use the standard pattern-card contract:

- `## Summary`
- `## Use When`
- `## Avoid When`
- `## Signals`
  - `### Code`
  - `### Profile`
  - `### IR`
- `## What To Verify After Applying`
- `## Related Patterns`

Keep the rest of the user's material in free-form sections such as `## Problem`, `## Core Abstraction`, `## Optimization Strategy`, `## Implementation Sketch`, and `## Pitfalls / Risks`.

### Naming

Use the stable identifier `reduce-avoid-transpose-copy` and a human title that makes the non-last-dimension reduction scenario obvious at a glance.

### Priority

Leave the card at normal priority so it shows up in the generated summary list without entering the `## High Priority Patterns` section.

## Validation

- Regenerate `skills/triton-npu-optimize-knowledge/references/pattern_index.md`.
- Run the pattern-index generator with `--check`.
- Re-read the generated summary entry to confirm the triage wording is concise and orthogonal to the full card.

## Scope Boundaries

- Do not hand-edit `pattern_index.md`.
- Do not change the pattern-index generator.
- Do not modify symptom cards.
