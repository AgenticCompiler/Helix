# Optimize Pattern Card Authoring

The Markdown files under `skills/triton/triton-npu-optimize-knowledge/references/patterns/` are the authored source of truth for generic optimize pattern knowledge.

Do not hand-edit `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`. It is generated from the pattern cards in that directory.

## Authoring Contract

Each pattern card may contain:

- optional frontmatter metadata used by the generator
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
`## Avoid When`, `## Signals` (and its `### Code`, `### Profile`, `### IR` subsections), `## What To Verify After Applying`, and `## Related Patterns` are also kept in the authored card only; they should not be emitted into the generated `pattern_index.md`, because the index is for triage rather than detailed detection or validation detail.
The generated `pattern_index.md` also includes a `## High Priority Patterns` section that lists only cards marked `priority: high`.

## Practical Rules

- Every pattern card should begin with a top-level `# <Human Title>` heading.
- Every pattern card must include both `## Summary` and `## Use When`.
- Pattern-card frontmatter may include `priority: high|normal`.
- If omitted, the generator will default to `normal`.
- `priority` only affects generated index presentation.
- **`## Summary` describes WHAT the pattern is/does.** Keep it to 1-2 sentences. Do not include signal-like language ("when X happens", "look for"), usage instructions, or implementation detail.
- **`## Use When` describes WHEN to apply the pattern.** Keep it as detection conditions. Do not describe what the pattern is.
- **Summary and Use When must be orthogonal**: information in one should not appear in the other.
- Missing optional predefined sections are allowed.
- `## Use When`, `## Avoid When`, and the optional signal subsections work best as bullet lists.
- Both `- item` and `1. item` list styles are accepted by the generator.
- Keep `pattern_index.md` generated and checked in with the authored cards.

## Regenerating The Index

After editing any pattern card, regenerate the checked-in index:

```bash
uv run python -m triton_agent.optimize_knowledge.pattern_index \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  --style default
```

To verify that the checked-in index is up to date without rewriting it:

```bash
uv run python -m triton_agent.optimize_knowledge.pattern_index \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  --style default \
  --check
```
