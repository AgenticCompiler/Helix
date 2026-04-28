# Optimize CANN Extension API Gating Implementation Plan

## Goal

Implement `--enable-cann-ext-api` for optimize workflows, gated to `A5`, with conditional skill staging and prompt exposure.

## Steps

1. Add failing tests for CLI parsing and option mapping.
2. Add failing tests for optimize validation rejecting `--enable-cann-ext-api` on non-`A5` targets.
3. Add failing tests for optimize request staging and prompt text when the feature is enabled or disabled.
4. Implement the new CLI option, runtime option field, and validation rule.
5. Implement conditional prompt text and conditional staged skill inclusion.
6. Add the dedicated skill directory with the specialized pattern reference.
7. Update README and contract-style tests that assert documented optimize options.
8. Run targeted tests, then repository verification for touched areas.
