# Orchestration Module Rename Design

## Summary

- Rename `src/helix/optimize/orchestration.py` to `src/helix/optimize/orchestration.py`.
- Rename `src/helix/generation/orchestration.py` to `src/helix/generation/orchestration.py`.
- Update in-repo imports and tests so package names describe orchestration responsibilities more clearly.

## Goals

- Replace the vague `runtime.py` name with a filename that matches what these modules actually do.
- Keep optimize and generation package naming consistent with each other.
- Preserve all current request-building and request-running behavior.

## Non-Goals

- Do not split request-building into new modules in this change.
- Do not change prompts, skill staging, runner behavior, or CLI semantics.
- Do not rewrite historical design docs just to remove stale path mentions.

## Design

- Move the optimize module to `optimize/orchestration.py` without changing exported function names.
- Move the generation module to `generation/orchestration.py` without changing exported function names.
- Update code and test imports to use the new module paths.
- Remove the old `runtime.py` module paths instead of leaving compatibility shims.

## Verification

- Run `uv run python -m unittest tests.test_generation_commands tests.test_optimize_runtime tests.test_cli -v`
- Run `uv run --group dev ruff check`
- Run `uv run pyright`
