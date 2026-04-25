# Pyright Strict Scope

## Summary

Keep repository-wide Pyright analysis enabled, apply strict type checking only to `src/`, keep `tests/` at the default basic level, and keep `skills/*/scripts/` at the default basic level.

## Rationale

Pyright does not offer a clean directory-level downgrade from a global `strict` baseline back to `basic` for `tests/`. The supported incremental path is to keep `typeCheckingMode = "basic"` and opt specific paths into strict checking with the `strict = [...]` list.

## Intended Behavior

- `uv run pyright` analyzes `src/`, `tests/`, and `skills/*/scripts/`.
- Files under `src/` are checked in strict mode.
- Files under `tests/` remain at the default basic mode unless explicitly promoted later.
- Files under `skills/*/scripts/` remain at the default basic mode unless explicitly promoted later.

## Verification

- Update the repository contract test to assert the `pyproject.toml` Pyright settings.
- Run `uv run pyright` and fix any new findings under `src/`, `tests/`, or `skills/*/scripts/`.
