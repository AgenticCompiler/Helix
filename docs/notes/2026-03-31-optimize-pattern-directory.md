## Summary

- Move optimize pattern references into a dedicated `references/patterns/` directory.
- Keep workflow and contracts at the top level of `references/` while treating patterns as a separate drill-down library.

## User-Visible Behavior

- The optimize skill should point to `references/pattern_index.md` as the entry point for pattern selection.
- Detailed pattern references should live under `references/patterns/` with shorter names such as `tiling.md` and `reorder-load.md`.

## Implementation Notes

- Update all links that previously pointed at `pattern-index.md` or `pattern_*.md`.
- Keep `references/knowledge/` available for future background material, but do not force it into the main optimize workflow yet.
