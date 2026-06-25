# Convert Prompt Triton Continuity Design

## Summary

- Tighten the `convert` skill and CLI prompt so the workflow explicitly requires a real Triton kernel implementation rather than a pure PyTorch rewrite.
- Keep this iteration limited to prompt and contract wording only.
- Do not add runtime enforcement, static output checks, or new staged skills in this change.

## Problem

- The current `convert` workflow already says the output should be Triton NPU-backed, but the constraint is too soft.
- In practice an agent can still interpret the task as "make the outputs match" and return a pure PyTorch implementation that passes differential validation.
- That outcome violates the user-visible purpose of `convert`, which is to produce a Triton-backed operator artifact, not just any behaviorally equivalent rewrite.

## Goals

- State more plainly that the converted artifact must contain and use a Triton kernel path for the converted computation.
- Make the prohibition against pure PyTorch substitution prominent in both the skill contract and the generated CLI prompt.
- Preserve the existing flexibility that the public API may remain PyTorch-facing.
- Add regression tests that pin the stronger wording.

## Non-Goals

- Do not add post-run validation that inspects the converted file for Triton launch signals.
- Do not stage `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` or reuse optimize round checks in `convert`.
- Do not change convert execution flow, output naming, or differential-test workflow.

## Recommended Approach

- Update `skills/triton/triton-npu-convert-pytorch-operator/SKILL.md` so the required workflow and quality rules explicitly say:
  - the task is to implement the converted computation as a Triton Ascend NPU kernel path
  - keeping a PyTorch-facing wrapper or module API is fine when that is the intended public entrypoint
  - a pure PyTorch rewrite does not satisfy the convert contract, even if differential tests pass
- Update the `CommandKind.CONVERT` branch in `src/triton_agent/prompts.py` with matching wording so every backend receives the same stronger policy.
- Add focused prompt and contract tests to prevent later wording regressions.

## Why Prompt-Only First

- This is the smallest change that directly addresses the reported agent behavior.
- It matches the requested first step before introducing stronger enforcement.
- It avoids false positives from a new static checker while still improving the workflow contract immediately.

## Testing

- Extend convert prompt tests in `tests/test_cli.py` to assert the stronger Triton-kernel requirement and pure-PyTorch prohibition.
- Extend generation contract tests in `tests/test_generation_contracts.py` to assert the convert skill documents the same rule.

## Expected Outcome

- Convert prompts will describe the target more precisely: a PyTorch-facing operator backed by real Triton kernel code.
- Agents will receive a clearer instruction that "correct outputs" alone are not enough if the implementation falls back to pure PyTorch.
- The repository will have regression coverage for this prompt policy while leaving room for a later hard gate if needed.
