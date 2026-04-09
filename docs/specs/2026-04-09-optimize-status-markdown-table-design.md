# `optimize-status` Markdown Table Output Design

## Summary

- Add an alternate `optimize-status` output mode that renders a compact Markdown table.
- Keep the existing text output as the default.
- Restrict the Markdown table to workspaces with optimize-session artifacts and summarize only the two speedup metrics.

## CLI

- Add `--format` to `optimize-status` with choices:
  - `text` (default)
  - `markdown`

## Markdown Table Behavior

- Output only one Markdown table with columns:
  - `名称`
  - `Geomean speedup`
  - `Total speedup`
- Exclude `no-session` workspaces from the table.
- Include `warning` and `ok` workspaces.
- Render unavailable speedup values as `-`.
- Keep the existing workspace ordering semantics:
  - `warning` before `ok`
  - name ordering within each state

## Non-Goals

- Do not change optimize-status analysis, warnings, or metric computation.
- Do not mix the Markdown table with the existing multi-line text rendering in the same output mode.
