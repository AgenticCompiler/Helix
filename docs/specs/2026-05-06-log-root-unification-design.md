# Log Root Unification Design

## Summary

Keep `triton-agent-logs/` as the only top-level workspace log directory.

## Context

The CLI now writes shared code-agent `--show-output` logs under `triton-agent-logs/`, while optimize session archives still use a separate `optimize-logs/` tree. That split makes the workspace log layout harder to explain and leaves two top-level directories for closely related runtime artifacts.

## Decision

Move optimize session archives under `triton-agent-logs/triton-agent/` and stop creating `optimize-logs/`.

This keeps one shared workspace log root while preserving a dedicated optimize archive namespace inside it.

## Scope

- Update optimize archive path construction.
- Update optimize reset cleanup to remove only `triton-agent-logs/triton-agent/`.
- Update tests and current-behavior docs to describe the unified log root.

## Non-Goals

- Do not rename `--show-output` log files.
- Do not redesign optimize archive contents.
- Do not delete unrelated files under `triton-agent-logs/`.

## Reset Behavior

`--reset-optimize` must continue to remove only recognized optimize-session artifacts. After this change, that means cleaning the optimize archive subtree under `triton-agent-logs/triton-agent/` rather than deleting the whole shared log root.

## Verification

- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`
