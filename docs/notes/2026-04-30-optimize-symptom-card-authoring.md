# Optimize Symptom Card Authoring

The Markdown files under `skills/triton-npu-optimize-knowledge/references/symptoms/` are the authored source of truth for generic optimize symptom knowledge.

Do not hand-edit `skills/triton-npu-optimize-knowledge/references/symptom_index.md`. It is generated from the symptom cards in that directory.

## Authoring Contract

Each symptom card must include:

- `## Summary`
- `## Evidence To Confirm`
- `## Candidate Pattern Directions`

Each symptom card may additionally include:

- `## Common Non-Matches`

## Regenerating The Index

```bash
python3 skills/triton-npu-optimize-knowledge/scripts/build_symptom_index.py \
  --symptoms-dir skills/triton-npu-optimize-knowledge/references/symptoms \
  --output skills/triton-npu-optimize-knowledge/references/symptom_index.md
```
