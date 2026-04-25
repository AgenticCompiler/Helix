## Summary

- Extend the optimize pattern index so newly added pattern files participate in pattern selection instead of remaining undiscoverable.

## User-Visible Behavior

- The optimize skill should expose all current pattern references through `references/patterns/index.md`.
- The detailed pattern list in `skills/triton-npu-optimize/SKILL.md` should match the actual files present under `references/patterns/`.

## Implementation Notes

- Add concise selection summaries for the new pattern files.
- Keep the index short enough to support selective reading rather than recreating the full detailed pattern content.
