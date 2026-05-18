# Optimize Pattern Priority Design

## Summary

Add an optional per-pattern priority flag so the generated optimization pattern index can highlight the small set of patterns that should be surfaced first.

Keep the authored pattern cards as the source of truth, and keep priority as lightweight metadata rather than a new content section.

## Goal

Make it possible to mark individual optimize pattern cards as high priority and have the generated index render a dedicated high-priority pattern list before the full summary list.

## User-Visible Behavior

- Pattern cards may declare `priority: high` or `priority: normal` in frontmatter.
- If a pattern card omits `priority`, the generator treats it as `normal`.
- The generated pattern index adds a `## High Priority Patterns` section before `## Generated Pattern Summaries`.
- The high-priority section lists only patterns marked `high`.
- High-priority patterns still appear in the full generated summary list.
- For the initial rollout, only `autotune` and `a5-force-simt-only-discrete-access` are marked `high`.

## Design

### Authoring Contract

Priority belongs in pattern-card frontmatter because it is metadata about how the index should present the card, not authored optimization guidance content.

Accepted values are:

- `high`
- `normal`

If a card provides any other value, index generation must fail explicitly instead of silently coercing or ignoring it.

Cards without frontmatter remain valid. Cards with frontmatter are not required to declare `priority`.

### Index Rendering

The generated pattern index keeps its current full-summary section and adds one compact section ahead of it:

```text
## High Priority Patterns
```

Each entry in that section should include:

- the pattern identifier
- a source link to the pattern card
- the one-line summary

This section is intentionally compact and does not expand `Use When`, because it is meant to be a first-pass shortlist rather than a replacement for the full index.

If no cards are marked `high`, the section should still be rendered with a stable placeholder line such as `- None.` so the generated file shape remains predictable.

### Sorting And Compatibility

Pattern priority should not change the existing ordering of the full generated summary list.

Priority only controls whether a pattern is also repeated in the dedicated high-priority section. This avoids changing the current browsing flow for readers who already rely on the full index ordering.

## Validation

- Regenerate `skills/triton-npu-optimize-knowledge/references/pattern_index.md`.
- Run the pattern-index generator with `--check` to confirm the checked-in file is current.
- Run the file-scoped strict pyright check for `skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`.

## Scope Boundaries

- Do not add a new required Markdown section to pattern cards.
- Do not bulk-edit every pattern card to add `priority: normal`.
- Do not remove high-priority patterns from the full summary section.
- Do not change symptom-card behavior or symptom-index generation.
