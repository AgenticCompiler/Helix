## Summary

Simplify backend launch lifecycle by keeping a single backend preparation context in `AgentRunner`, instead of splitting backend setup across `_prepare_run_context()` and `_prepare_agent_hooks_context()`.

## User-Visible Semantics

- Backend behavior stays the same.
- MCP staging, hook staging, logging, retries, tracing, and cleanup semantics stay the same.

## Design

- `AgentRunner.run()` uses only `_prepare_run_context()`.
- Backend subclasses own all backend-specific preparation inside that single context, including MCP config staging and hook staging.
- Base keeps only shared helpers such as hook option construction and extra allowed read roots.

## Constraints

- Do not change current file paths, policy values, or cleanup ordering.

## Verification

- Update base/backend tests to match the single-context lifecycle.
- Run focused backend tests plus repository lint, type check, and full test suite.
