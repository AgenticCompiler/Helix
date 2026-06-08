# Remove Optimize Continuous Runtime Design

## Goal

Remove optimize's legacy `continuous` runtime path completely so optimize only supports the batched `checked` and `supervised` flows.

## User-Visible Semantics

- Optimize requests only run through the batched multi-invocation controller.
- Any remaining internal/runtime-only `continuous` optimize mode support is deleted rather than preserved as compatibility behavior.
- Prompt builders and request models no longer describe optimize as a continuous session mode.

## Implementation Notes

- Delete `execute_continuous_optimize()` dispatch from optimize orchestration and execution.
- Narrow optimize round-mode literals in request/option models to `checked | supervised`.
- Remove continuous-only optimize prompt helpers and tests.
- Keep generic non-optimize uses of words like "continuous" in knowledge docs untouched.
