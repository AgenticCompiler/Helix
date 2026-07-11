# Verification Command Rename Design

## Summary

- Rename the user-facing commands `optimize-verify` and `optimize-verify-batch` to `verify` and `verify-batch`.
- Do not keep compatibility aliases for the old command names or old snake_case spellings.
- Move verification implementation out of `src/helix/optimize/` into a dedicated `src/helix/verification/` package.
- Present verification as its own CLI help group instead of listing it under optimization.

## Goals

- Make the CLI surface match the real responsibility of the feature: verification, not optimization orchestration.
- Make module boundaries match the command surface so verification code no longer lives under `optimize/`.
- Preserve the current verification behavior, on-disk artifacts, and remote execution semantics while renaming the feature.

## Non-Goals

- Do not change verification result formats such as `opt-verify/verify-*/verify-state.json`.
- Do not change optimize status semantics beyond updating references to the renamed verification command.
- Do not keep deprecated aliases or compatibility shims for the old command names or old module paths.

## User-Facing Behavior

- `uv run helix verify -i .`
- `uv run helix verify -i . --phase test`
- `uv run helix verify -i . --remote alice@example.com`
- `uv run helix verify-batch -i operators_root`
- `uv run helix verify-batch -i operators_root --force-verify`

The old commands:

- `optimize-verify`
- `optimize-verify-batch`

must stop working instead of acting as hidden aliases.

## CLI Structure

- Replace `CommandKind.OPTIMIZE_VERIFY` with `CommandKind.VERIFY`.
- Replace `CommandKind.OPTIMIZE_VERIFY_BATCH` with `CommandKind.VERIFY_BATCH`.
- Add a new top-level help group named `Verification`.
- Move `verify` and `verify-batch` into that group.
- Remove old alias normalization entries for `optimize_verify` and `optimize_verify_batch`.
- Add new alias normalization entries for `verify_batch` only if the repository still wants snake_case support for canonical command names. The old optimize-prefixed aliases must be removed.

## Module Boundaries

Create a dedicated package:

- `src/helix/verification/__init__.py`
- `src/helix/verification/core.py`
- `src/helix/verification/batch.py`

Move verification-specific types and helpers there:

- `VerifyOptions`
- `VerifyTarget`
- `VerifyResult`
- `prepare_verify_target()`
- `run_verify()`
- `run_verify_batch()`

Create a dedicated command entrypoint module:

- `src/helix/commands/verification.py`

`src/helix/commands/optimize.py` should no longer own verify command handlers.

## Behavior Preservation

Even after the rename:

- `verify` still validates the current best optimize round.
- `verify-batch` still operates on optimize workspaces under a root directory.
- `status` still reads the latest verification state from `opt-verify/verify-*/verify-state.json`.
- remote flags for `verify-batch` still apply uniformly to every workspace in the batch run.

## Tests And Docs

Update tests to assert:

- new command names parse and dispatch correctly
- old command names are absent from help output
- the new `Verification` help group is rendered
- renamed verification modules and handlers are imported from their new package locations

Update user-facing docs and design docs so they use `verify` and `verify-batch` terminology consistently.
