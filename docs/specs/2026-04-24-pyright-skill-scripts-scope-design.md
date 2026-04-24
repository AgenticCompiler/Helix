# Pyright Skill Scripts Scope Design

## Goal

- Keep the repository-wide `pyright` contract small and explicit.
- Add `skills/*/scripts/` Python helpers to the default `pyright` run.

## User-Visible Behavior

- `uv run pyright` continues to analyze `src/` and `tests/`.
- `uv run pyright` also analyzes Python files under `skills/*/scripts/`.
- Files under `src/` remain in strict mode.
- Files under `tests/` remain in basic mode.
- Files under `skills/*/scripts/` remain in basic mode.

## Non-Goals

- Do not promote skill scripts to strict mode in this change.
- Do not broaden `pyright` to all files under `skills/`.

## Verification

- Update the repository contract test for the `pyright` include/exclude rules.
- Run `uv run pyright`.
