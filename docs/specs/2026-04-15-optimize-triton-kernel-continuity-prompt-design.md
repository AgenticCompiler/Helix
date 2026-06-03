# Optimize Triton Kernel Continuity Prompt Design

## Summary

- Tighten the optimize worker and supervisor prompts so `optimize` no longer treats a pure PyTorch rewrite as a successful operator optimization.
- Keep the existing public-entrypoint flexibility: a valid operator may still expose a PyTorch-facing wrapper function or module class.
- Define the optimization target more precisely in prompt wording: the round must continue optimizing the same Triton Ascend NPU kernel path instead of replacing core computation with a pure PyTorch implementation.
- Limit this change to prompt guidance and prompt tests for now; do not add new `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` enforcement in this iteration.

## Problem

- The repository already supports generated test and benchmark harnesses whose public API may be a Triton wrapper, a PyTorch function, or a `torch.nn.Module`.
- That flexibility is correct for generation and execution, but it leaves optimize runs vulnerable when the worker interprets "faster correct output" as permission to bypass the Triton kernel entirely.
- In practice this allows a cheating failure mode: the agent can replace the operator's core implementation with direct PyTorch ops, still pass correctness and even produce benchmark numbers, yet completely miss the actual task of optimizing the Triton Ascend NPU operator.
- The current optimize prompts emphasize artifacts, evidence, and `compare-perf`, but they do not explicitly forbid this kernel-bypass behavior.

## Goals

- Make optimize prompts explicitly state that the task is to optimize the Triton Ascend NPU operator implementation itself.
- Allow a PyTorch-facing public API to remain as a wrapper when that matches the operator's intended interface.
- Explicitly forbid replacing the operator's core Triton kernel path with a pure PyTorch implementation.
- Teach supervised optimize audits to reject rounds that appear to have bypassed the Triton kernel.
- Add prompt-level regression tests so later refactors do not silently weaken this guidance.

## Non-Goals

- Do not add new `baseline/state.json` or `round-state.json` fields in this change.
- Do not modify `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` or add static or runtime kernel continuity checks yet.
- Do not change generation, test, or benchmark harness contracts.
- Do not forbid PyTorch wrappers or modules as public operator entrypoints in general.

## Approaches Considered

### Recommended: Prompt-Only Policy Tightening

- Update optimize worker, unsupervised, and supervisor prompts with explicit kernel continuity language.
- Add tests that pin the new wording in prompt builders and supervised runtime behavior.

Why this is the best fit now:

- It addresses the reported failure mode with the smallest possible change.
- It avoids expanding the optimize artifact contract before confirming whether stronger enforcement is necessary.
- It preserves the team's preferred rollout order: try clearer prompt constraints first, then escalate to checks only if needed.

### Alternative: Add `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` Enforcement Immediately

- Extend baseline and round contracts with canonical kernel identity and continuity fields.
- Fail rounds whose artifacts suggest the Triton kernel was bypassed.

Why not choose this now:

- It is heavier than the requested first step.
- It risks false positives if the initial continuity heuristic is incomplete.
- It changes optimize metadata contracts, which should be justified by observed prompt-level insufficiency.

### Alternative: Require Runtime Profiler Evidence For Every Round

- Require profile or msprof evidence proving the target kernel executed in each accepted round.

Why not choose this now:

- It adds significant execution cost and operational complexity.
- It is a better fit for a future strict mode than for the default optimize path.

## User-Facing Design

### Worker And Unsupervised Prompt Policy

Optimize prompts should say all of the following plainly:

- the task is to optimize the Triton Ascend NPU operator implementation itself
- a PyTorch-facing public API may remain as a wrapper when that is the intended operator entrypoint
- the worker must continue reusing and improving the Triton kernel path rather than replacing core computation with direct PyTorch ops
- a round that swaps the kernel path for a pure PyTorch implementation does not count as a successful optimization round

This wording should appear in both:

- the supervised worker prompt built by `build_optimize_worker_prompt()`
- the unsupervised optimize prompt built by `build_optimize_unsupervised_prompt()`

### Supervisor Audit Policy

The supervisor prompt should explicitly audit for the same policy:

- if the latest round appears to keep only the public API shape while bypassing the Triton kernel with pure PyTorch computation, the supervisor should reject the round and require revision

This keeps the role separation intact:

- workers perform the round
- supervisors do not edit the operator
- supervisors still decide whether the produced round satisfies the optimize contract

## Architecture

### Prompt Construction

- Keep the change localized to prompt-building helpers in `src/triton_agent/prompts.py`.
- Reuse the existing optimize prompt builders instead of introducing a second policy layer.
- Preserve all current artifact, baseline, evidence, and `compare-perf` instructions.
- Add the new kernel continuity language as a small cluster of adjacent lines so future maintainers can reason about the rule as one policy.

### Runtime Behavior

- No optimize runtime control-flow change is required for this version.
- Existing supervised and unsupervised flows already pass through the prompt builders, so stronger wording there is sufficient for the first rollout.

### Future Escalation Path

If prompt tightening is not enough, the next iteration should add hard enforcement by:

- recording canonical kernel identity in baseline metadata
- recording round-level kernel continuity metadata
- teaching `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` to reject rounds that preserve only output equivalence while bypassing the Triton kernel path

This document intentionally leaves that follow-up as future work.

## Testing

- Add prompt tests that assert worker prompts include the new kernel continuity wording.
- Add prompt tests that assert unsupervised optimize prompts include the same restriction.
- Add supervisor prompt tests or supervised runtime tests that assert supervisor audits mention rejecting pure PyTorch substitution.
- Keep the tests narrow and text-focused so they protect the policy without over-coupling to unrelated prompt wording.

## Documentation

- No README update is required for this iteration because the change affects optimize policy wording rather than user-facing CLI flags or workflow steps.
- The spec and implementation plan should record that this is an intentional first step before adding enforcement checks.

## Expected Outcome

- Optimize agents receive an explicit instruction that "faster correct outputs" are insufficient when achieved by replacing the Triton kernel with direct PyTorch code.
- Supervised audits gain explicit wording to reject kernel-bypass rounds.
- The project gets low-cost coverage against regression in this policy area while keeping room to add hard checks later if real runs still cheat.
