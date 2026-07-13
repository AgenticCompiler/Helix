# Verbose Hooks Color Design

## Goal

Make the `[hooks]` verbose prefix render with ANSI color in TTY output, matching the existing behavior of other verbose categories.

## Problem

The backend runtime emits verbose lines under the `hooks` category, but `src/helix/verbose.py` does not assign a color for that category. As a result, `[hooks]` falls back to plain text while nearby categories such as `[command]`, `[skills]`, and `[remote]` are colorized.

## User-Visible Behavior

- TTY verbose output should colorize `[hooks]`.
- Non-TTY redirected output should remain plain text.
- Existing colors for other verbose categories should remain unchanged.

## Design

- Add a `hooks` entry to the shared `COLORS` map in `src/helix/verbose.py`.
- Reuse the existing hook/log prefix formatter without introducing a hooks-specific output path.

## Testing

- Add a focused TTY unit test that verifies `emit_verbose(..., "hooks", ...)` emits a colorized prefix.
- Keep the existing command and agents color tests unchanged.

## Scope

- Do not change hook message text.
- Do not change non-hooks verbose formatting.
