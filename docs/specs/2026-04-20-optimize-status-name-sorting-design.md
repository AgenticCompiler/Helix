# Optimize Status Name Sorting Design

## Summary

- Sort Markdown `optimize-status` rows by workspace/operator name.
- Sort text `optimize-status` output with `NO-SESSION` workspaces first, then all remaining workspaces by name.
- Stop grouping non-`NO-SESSION` text output by `WARN` before `OK`.

## Goals

- Make copied Markdown tables easier to scan by operator name.
- Keep `NO-SESSION` workspaces prominent in text output.
- Make the remaining text output stable and alphabetical regardless of status.

## Non-Goals

- Do not change status calculation.
- Do not change per-workspace fields or warning text.
- Do not change batch optimize result ordering.

## User-Visible Behavior

Text output order:

1. All `NO-SESSION` workspaces, sorted by name.
2. All remaining workspaces, sorted by name regardless of `WARN` or `OK`.

Markdown output order:

1. Exclude `NO-SESSION` workspaces.
2. Sort every remaining row by name regardless of status.

## Implementation Shape

- Replace the single status-priority sort helper with render-specific helpers.
- Use one helper for text output and one helper for Markdown output.
- Update render and CLI tests with names that distinguish alphabetical ordering from status grouping.

## Testing

- Render test proving text output keeps `NO-SESSION` first and then sorts `OK`/`WARN` by name.
- CLI test proving the same text ordering.
- Markdown render and CLI tests proving rows are sorted by name regardless of status.
