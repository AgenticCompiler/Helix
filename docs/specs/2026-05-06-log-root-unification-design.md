# Log Root Unification Design

## Summary

Keep `helix-logs/` as the only top-level workspace log directory.

## Context

The CLI now writes shared code-agent `--show-output` logs under `helix-logs/`, while optimize session archives still use a separate `optimize-logs/` tree. That split makes the workspace log layout harder to explain and leaves two top-level directories for closely related runtime artifacts.

## Decision

Move optimize session archives under `helix-logs/helix/` and stop creating `optimize-logs/`.

This keeps one shared workspace log root while preserving a dedicated optimize archive namespace inside it.

## Scope

- Update optimize archive path construction.
- Update optimize reset cleanup to remove only `helix-logs/helix/`.
- Update tests and current-behavior docs to describe the unified log root.

## Non-Goals

- Do not rename `--show-output` log files.
- Do not redesign optimize archive contents.
- Do not delete unrelated files under `helix-logs/`.

## Reset Behavior

`--reset-optimize` must continue to remove only recognized optimize-session artifacts. After this change, that means cleaning the optimize archive subtree under `helix-logs/helix/` rather than deleting the whole shared log root.

## Verification

- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`
