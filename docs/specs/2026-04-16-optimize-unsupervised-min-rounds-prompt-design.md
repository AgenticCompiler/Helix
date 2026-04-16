# Optimize Unsupervised Min Rounds Prompt Design

## Summary

- Teach the unsupervised optimize prompt to mention `--min-rounds` when the user sets it.
- Keep supervised worker prompts unchanged: worker rounds still should not decide session-level minimum-round policy.
- Preserve the existing runtime enforcement that resumes an unsupervised run when the agent exits before enough `opt-round-*` directories exist.

## Problem

- `build_optimize_unsupervised_prompt()` currently accepts `min_rounds` but discards it.
- As a result, the first unsupervised agent invocation does not know the session has a minimum round requirement.
- The runtime loop still enforces `min_rounds` after the agent exits, but that is later and weaker than telling the agent up front.

## Goals

- Make the first unsupervised optimize prompt explicitly state the required minimum number of optimization rounds when configured.
- Avoid changing supervised worker semantics.
- Keep the change limited to prompt wording and prompt tests.

## Non-Goals

- Do not change `OptimizeRunLoop` minimum-round enforcement.
- Do not add new optimize metadata fields or round-contract checks.
- Do not change supervised worker or supervisor prompt ownership boundaries.

## Design

### Unsupervised Prompt

When `min_rounds` is set, `build_optimize_unsupervised_prompt()` should add a clear session-level instruction such as:

- complete at least `N` optimization rounds before deciding the session should stop
- once that minimum is satisfied, stop after the current passing round unless there is a concrete reason to continue

This line belongs near the existing unsupervised session-ownership language so the agent sees it as part of the top-level stopping rule.

### Supervised Prompt Boundary

Keep `build_optimize_worker_prompt()` unchanged with respect to `min_rounds`.

- supervised workers still own exactly one round
- the orchestration loop still decides whether more rounds are required

## Testing

- Add a failing prompt test that asserts unsupervised optimize prompts mention the configured minimum round count.
- Keep the existing worker test that asserts supervised worker prompts do not mention `min_rounds`.
