# Centralize Optimize Worker Prompt Construction

## Summary

- Make `build_optimize_request` produce optimize request metadata only.
- Stop precomputing the first worker prompt in orchestration.
- Build every worker prompt, including the first batch, inside the optimize execution loop.

## Problem

- `build_optimize_request` currently computes an initial optimize worker prompt and stores it in `AgentRequest.prompt`.
- `MultiInvocationOptimizeController` then rebuilds later worker prompts inside `execution.py`.
- That splits one worker-prompt contract across two modules and makes the first batch behave differently from later batches.

## Decision

- `build_optimize_request` should still resolve:
  - resume state
  - output path
  - batch bounds
  - staged skills
  - MCP/env metadata
  - compiler-source metadata
- But it should no longer build a worker prompt up front. Set `request.prompt` to an empty string and keep raw user input in `request.user_prompt`.
- `execution.py` becomes the single owner of worker prompt construction:
  - first batch worker prompt
  - later batch worker prompt
  - previous-batch follow-up injection
- Baseline repair prompt should use explicit request metadata plus `request.user_prompt`, not a prebuilt worker prompt.

## Expected Behavior

- First batch and later batches use the same worker-prompt builder.
- The only optimize prompts built outside the worker loop remain:
  - baseline repair prompt
  - supervisor prompt
- User-provided `--prompt` guidance still appears in:
  - the baseline repair prompt
  - the first worker prompt
  - later worker prompts

## Verification

- Update tests so `build_optimize_request` no longer exposes a prebuilt worker prompt.
- Add regression coverage that the first worker batch prompt is built in `execution.py` from request metadata and user instructions.
