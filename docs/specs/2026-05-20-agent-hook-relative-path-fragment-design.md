# Agent Hook Relative Path Fragment Design

## Goal

Make the opt-in Codex and OpenCode hook guards accept documented staged helper
script entrypoints when the command uses relative backend-native paths such as
`.opencode/skills/.../scripts/run-command.py`.

## Root Cause

Both guards already allow a protected staged script when it is the Python
entrypoint token. The remaining false positive comes from the fallback
path-fragment scan: it also matches nested slash-delimited suffixes inside that
same token, such as `/skills/...` and `/scripts/...`, and then misclassifies
those substrings as absolute paths outside the workspace.

## Design

- Keep the existing fallback path-fragment scan for Python one-liners and other
  cases where a path only appears inside a larger token.
- When a full path-like token has already been collected from shell tokenization,
  ignore any fallback regex match that is only a nested substring of that token.
- Preserve the existing protected-script entrypoint exception and keep direct
  reads, Python `open(...).read()`, and out-of-workspace paths denied.

## Scope Boundaries

- Do not relax protection for staged skill scripts beyond documented execution
  entrypoints.
- Do not remove the fallback fragment scan entirely.
- Do not change hook staging, runner flags, or denial text.
