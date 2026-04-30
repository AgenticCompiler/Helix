# Optimize Pattern Card Authoring

The Markdown files under `skills/triton-npu-optimize-knowledge/references/patterns/` are the authored source of truth for generic optimize pattern knowledge.

Do not hand-edit `skills/triton-npu-optimize-knowledge/references/pattern_index.md`. It is generated from the pattern cards in that directory.

## Authoring Contract

Each pattern card may contain:

- predefined sections used by the generator
- free-form sections for additional explanation, examples, or architecture notes

Each pattern card should begin with a top-level `# <Human Title>` heading before the predefined sections.

The generator recognizes these predefined sections:

- required:
  - `## Summary`
  - `## Use When`
- optional:
  - `## Avoid When`
  - `## Signals`
  - `## Related Patterns`
  - `## What To Verify After Applying`

Inside `## Signals`, the generator also recognizes these optional subsections:

- `### Code`
- `### Profile`
- `### IR`

Free-form sections are allowed and stay in the authored card, but they are ignored for first-layer index generation.
`## What To Verify After Applying` and `## Related Patterns` are also kept in the authored card only; they should not be emitted into the generated `pattern_index.md`, because the index is for triage rather than post-apply validation or cross-pattern navigation detail.

## Practical Rules

- Every pattern card should begin with a top-level `# <Human Title>` heading.
- Every pattern card must include both `## Summary` and `## Use When`.
- Missing optional predefined sections are allowed.
- `## Use When`, `## Avoid When`, and the optional signal subsections work best as bullet lists.
- Both `- item` and `1. item` list styles are accepted by the generator.
- Keep `pattern_index.md` generated and checked in with the authored cards.

## Regenerating The Index

After editing any pattern card, regenerate the checked-in index:

```bash
python3 skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton-npu-optimize-knowledge/references/pattern_index.md
```

To verify that the checked-in index is up to date without rewriting it:

```bash
python3 skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton-npu-optimize-knowledge/references/pattern_index.md \
  --check
```
