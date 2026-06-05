## Summary

Finish the backend hook ownership refactor by removing the mixed shared/backend-specific `agent_hooks.py` module and splitting its contents into backend-local staging modules plus a minimal shared hook lifecycle helper.

## User-Visible Semantics

- Codex and OpenCode hook staging behavior remains unchanged.
- Hook guard and tool tracing still use the same files, paths, policy values, and cleanup behavior.
- Base runner behavior remains unchanged.

## Design

- Move shared hook lifecycle types and cleanup helpers into a small backend-shared helper module.
- Move Codex hook staging implementation into a Codex-local module.
- Move OpenCode hook staging implementation into an OpenCode-local module.
- Update tests to import the new backend-local helpers instead of the old mixed module.
- Remove `src/triton_agent/agent_hooks.py`.

## Constraints

- Do not rename the staged hook files or change their contents.
- Do not change current backend public behavior or request flags.

## Verification

- Run focused hook/backend tests first.
- Run repository lint, type check, and full test suite.
