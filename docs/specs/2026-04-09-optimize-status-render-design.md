# Optimize Status Render Design

## Summary

- Reorder `optimize-status` output so `NO-SESSION` workspaces print first, followed by all remaining workspaces sorted by name.
- Add lightweight ANSI styling in the render layer so titles and content are easier to scan in a terminal.
- Keep warning detail visually de-emphasized instead of making warning lines the loudest part of the report.

## Goals

- Surface untouched workspaces immediately when scanning large batch roots.
- Make per-workspace output easier to skim without changing the underlying status analysis.
- Keep redirected output plain text and stable for tests, logs, and copy-paste workflows.

## Non-Goals

- Do not change `optimize-status` workspace analysis or warning generation.
- Do not add new CLI flags for colors or sorting in this change.
- Do not colorize non-terminal output.

## User-Visible Behavior

### Ordering

- Results print in this order:
  1. `NO-SESSION` workspaces sorted alphabetically
  2. all remaining workspaces sorted alphabetically, regardless of `WARN` or `OK` state

### Styling

- Title lines such as `[WARN] layernorm` use an accent color.
- Detail lines such as `Baseline mean` and `Best round` use a softer body color.
- Warning detail lines such as `Warning: missing perf artifact for opt-round-28` use a faint gray so they stay visible without dominating the report.
- Summary output stays readable and understated.

### TTY Rules

- Emit ANSI color only when the target stream reports `isatty() == True`.
- When output is redirected, keep the exact plain-text structure without escape codes.

## Implementation Shape

- Keep sorting and color decisions inside `src/helix/optimize/render.py`.
- Add a small helper for text ordering that prioritizes `NO-SESSION` only.
- Add a small helper for TTY-aware styling instead of embedding escape codes directly in each `print`.
- Leave `src/helix/optimize/status.py` untouched unless tests reveal a coupling issue.

## Testing

- Add render-layer tests for `NO-SESSION`-first ordering followed by name sorting.
- Add render-layer tests proving non-TTY output stays plain text.
- Add render-layer tests proving TTY output applies accent styling to titles and faint styling to warnings.
