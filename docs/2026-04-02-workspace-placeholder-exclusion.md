# Workspace Placeholder Exclusion

## Summary

- Treat the top-level `workspace/` directory as a placeholder area for local experimentation.
- Do not rely on `workspace/` contents for repository tests, lint checks, or static type checks.

## User-Visible Behavior

- Repository verification commands should continue to validate `src/`, `tests/`, and documented skill assets.
- Placeholder files under `workspace/` may be added, removed, or changed without breaking repository verification.

## Implementation Notes

- Add `workspace/` to repository ignore rules.
- Exclude `workspace/` from Ruff and Pyright configuration.
- Replace tests that read `workspace/` sample files with checks against skill-owned reference content.
