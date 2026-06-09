---
name: create-optimize-pattern
description: Create new optimization pattern cards or symptom cards for the triton-agent optimize knowledge base. Use when adding a new pattern card under skills/triton-npu-optimize-knowledge/references/patterns/, skills/torch-npu-optimize-knowledge/references/patterns/, skills/triton-npu-cann-ext-api-patterns/references/patterns/, or a new symptom card under skills/triton-npu-optimize-knowledge/references/symptoms/.
---

# Create Optimize Pattern / Symptom Card

Use this skill when creating new optimization pattern cards or symptom cards. Do not use for editing existing cards — edit those directly.

## Target Directories

Place new cards under the appropriate directory:

| Card Type | Skill | Directory |
|---|---|---|
| Generic optimize pattern | `triton-npu-optimize-knowledge` | `skills/triton-npu-optimize-knowledge/references/patterns/` |
| Torch NPU optimize pattern | `torch-npu-optimize-knowledge` | `skills/torch-npu-optimize-knowledge/references/patterns/` |
| CANN extension API pattern | `triton-npu-cann-ext-api-patterns` | `skills/triton-npu-cann-ext-api-patterns/references/patterns/` |
| Generic optimize symptom | `triton-npu-optimize-knowledge` | `skills/triton-npu-optimize-knowledge/references/symptoms/` |

## Pattern Card Template

```md
---
priority: high|normal  # optional — omit to default to normal
---

# <Human Title>

## Summary

## Use When

## Avoid When

## Signals
### Code
### Profile
### IR

## Related Patterns

## What To Verify After Applying
```

### Pattern Card Rules

- **`## Summary` and `## Use When` are required.** All other predefined sections are optional.
- **`## Summary`** describes WHAT the pattern is/does. Keep it to 1-2 sentences. Do not include signal-like language or usage instructions.
- **`## Use When`** describes WHEN to apply the pattern — detection conditions. Must be orthogonal to `## Summary`.
- `priority: high|normal` in frontmatter; omit to default to `normal`. `priority` only affects generated index rendering. The generated `pattern_index.md` must include a `## High Priority Patterns` section that lists cards marked `high`.
- `## Use When`, `## Avoid When`, and signal subsections work best as bullet lists. Both `- item` and `1. item` list styles are accepted by the generator.
- Free-form sections are allowed for examples, background, or architecture notes — they are ignored by the index generator but kept in the authored card.

## Symptom Card Template

```md
# <Human Title>

## Summary

## Evidence To Confirm

## Candidate Pattern Directions

## Common Non-Matches
```

### Symptom Card Rules

- **`## Summary`, `## Evidence To Confirm`, and `## Candidate Pattern Directions` are required.**
- `## Common Non-Matches` is optional.
- Keep predefined section names exact.

## After Creation

1. Regenerate the affected index files:

```bash
bash scripts/update-optimize-knowledge-indices.sh
```

2. **Never hand-edit** `skills/*/references/pattern_index.md` or `skills/*/references/symptom_index.md` — they are generated.

## Reference

For detailed authoring notes, see `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`.
