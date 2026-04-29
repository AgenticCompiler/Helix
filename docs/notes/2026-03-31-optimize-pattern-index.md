## Summary

- Restructure the optimize skill knowledge base so the agent selects optimization patterns from a compact index before reading detailed pattern documents.
- Reduce the chance that long optimize runs load too many pattern files at once.

## User-Visible Behavior

- The optimize workflow should begin with pattern selection, not bulk pattern reading.
- The agent should read `references/pattern_index.md` first, then open only the one or two most relevant detailed pattern references for the current bottleneck.

## Implementation Notes

- Add a new `skills/triton-npu-optimize/references/pattern_index.md` file with concise summaries for each optimization pattern.
- Update `skills/triton-npu-optimize/SKILL.md` to require reading the pattern index before any detailed pattern file.
- Keep the detailed pattern files as the second-level drill-down references under `references/patterns/`.
