# Generation Package Refactor Design

## Summary

- Replace the flat `helix.generation` module with a `helix/generation/` package.
- Do not preserve the old `helix.generation` import path as a compatibility shim.
- Keep CLI behavior unchanged while making the generation code layout match the repository's current complexity more closely.

## Goals

- Make generation code easier to navigate by separating models, output-path handling, runtime orchestration, and batch orchestration.
- Bring generation structure closer to the clarity already present in `helix/optimize/` without forcing a one-to-one mirror.
- Keep the public CLI behavior for `gen-test`, `gen-bench`, `gen-eval`, and `gen-eval-batch` unchanged.

## Non-Goals

- Do not change prompt semantics, staged skill behavior, overwrite protection rules, or batch execution semantics.
- Do not add compatibility imports for `helix.generation`.
- Do not introduce extra generation submodules unless they own a clear responsibility today.
- Do not refactor unrelated optimize or execution code as part of this change.

## Proposed Package Shape

- `src/helix/generation/__init__.py`
  - Stable export surface for current generation helpers used elsewhere in the repository.
- `src/helix/generation/models.py`
  - `GenerationOptions`
- `src/helix/generation/outputs.py`
  - output path resolution
  - overwrite protection
  - target preparation helpers
- `src/helix/generation/orchestration.py`
  - request construction
  - staged skill selection
  - runner invocation
- `src/helix/generation/batch.py`
  - batch wrapper orchestration for `gen-eval-batch`

## Why This Shape

- `GenerationOptions` is a small domain model and should not live beside low-level runner orchestration.
- Output-path and overwrite logic are a separate concern from request construction and agent execution.
- `gen-eval-batch` is already conceptually a batch wrapper over the single-workspace generation runtime, so it belongs inside the generation package rather than as a top-level sibling module.
- Keeping the package to four focused modules avoids over-splitting the code just to mimic optimize's exact file layout.

## Import Migration

- Update all repository imports from:
  - `helix.generation`
  - `helix.generation_batch`
- Replace them with imports from the new package modules or from `helix.generation` package exports.
- Delete the old top-level files after all imports and tests are updated.

## User-Visible Semantics

- `gen-test`, `gen-bench`, `gen-eval`, and `gen-eval-batch` keep their current flags and defaults.
- Existing overwrite protection behavior remains unchanged.
- Existing `gen-eval-batch --input .` and directory-of-workspaces behavior remains unchanged.
- Result rendering and streaming output prefixes remain unchanged.

## Implementation Plan

- Create the `generation/` package and move the current implementation into focused modules.
- Re-export the required symbols from `generation/__init__.py` so internal call sites stay readable.
- Move `generation_batch.py` logic into `generation/batch.py`.
- Update command handlers and tests to import from the new package layout.
- Remove the old `generation.py` and `generation_batch.py` files once nothing imports them.

## Testing

- Keep current generation command tests and batch tests as regression coverage.
- Add or update import-level tests only as needed for the new package layout.
- Run at least:
  - `uv run python -m unittest tests.test_generation_commands tests.test_generation_batch tests.test_cli -v`
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
