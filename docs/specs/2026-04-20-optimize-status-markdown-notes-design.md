# Optimize Status Markdown Notes Design

## Summary

- Add a compact `Notes` column to `optimize-status --format markdown`.
- Use short labels instead of embedding detailed warning text in the table.
- Keep full warning detail in the default text output.

## Goals

- Make Markdown summaries show whether computed and logged best rounds disagree.
- Reserve room for future compact status annotations.
- Keep the table narrow enough for issue comments and copied reports.

## Non-Goals

- Do not add detailed round values to the Markdown `Notes` cell.
- Do not change the text output warning format.
- Do not change optimize-status analysis or best-round selection.

## User-Visible Behavior

Markdown output should include:

```markdown
| 名称 | Geomean speedup | Total speedup | Notes |
| --- | --- | --- | --- |
| matmul | 1.49x | 1.58x | best≠log |
| layernorm | - | - | warn |
| add | 1.25x | 1.25x | - |
```

Notes labels:

- `best≠log`: computed best round differs from the best round logged in `opt-note.md`.
- `warn`: other warnings exist for the workspace.
- `best≠log,warn`: both conditions apply.
- `-`: no compact notes.

## Implementation Shape

- Add a render-layer helper that derives a `Notes` cell from `OptimizeStatusWorkspace`.
- Treat the numeric/logged best mismatch warning as covered by `best≠log`, not as a separate `warn`.
- Update Markdown render and CLI tests.
- Update README Markdown output documentation.

## Testing

- Render test for `Notes` header, `warn`, `best≠log`, and `-`.
- CLI test for Markdown output including the new compact notes.
