# Pyright Strict Scope

## Summary

Keep repository-wide Pyright analysis enabled, but apply strict type checking only to `src/` while leaving `tests/` at the default basic level.

## Rationale

Pyright does not offer a clean directory-level downgrade from a global `strict` baseline back to `basic` for `tests/`. The supported incremental path is to keep `typeCheckingMode = "basic"` and opt specific paths into strict checking with the `strict = [...]` list.

## Intended Behavior

- `uv run pyright` analyzes both `src/` and `tests/`.
- Files under `src/` are checked in strict mode.
- Files under `tests/` remain at the default basic mode unless explicitly promoted later.

## Verification

- Update the repository contract test to assert the `pyproject.toml` Pyright settings.
- Run `uv run pyright` and fix any new strict findings under `src/`.
