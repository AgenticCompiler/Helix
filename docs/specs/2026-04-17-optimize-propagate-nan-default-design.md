# Optimize Propagate-NaN Default Design

## Summary

- Teach the optimize skill to treat explicit NaN propagation on Triton kernel compare helpers as an available consistency repair during optimization work.
- Let optimize agents add `propagate_nan=tl.PropagateNan.ALL` for kernel compare helpers such as `tl.maximum()` and `tl.minimum()` when those calls omit `propagate_nan`, not just the call sites touched by the current edit.
- Make the guidance explicit that this may change NaN-input behavior, so it should be treated as a semantic choice rather than a guaranteed no-op cleanup.
- Keep the change in skill guidance only; do not add CLI enforcement in this iteration.

## Problem

- Triton kernel optimization rounds may rewrite or otherwise preserve elementwise clamp or compare logic using compare helpers such as `tl.maximum()` and `tl.minimum()`.
- When any of those calls omit `propagate_nan`, the resulting NaN behavior is less explicit and may drift from the intended optimize policy.
- A rule that only covers newly added or directly edited calls would still leave older kernel call sites inconsistent within the same optimized operator.
- The current optimize skill does not call out this requirement, so an otherwise valid round can silently preserve or introduce ambiguous NaN propagation behavior.

## Goals

- Make the optimize skill explicitly call out NaN propagation semantics for kernel compare helpers such as `tl.maximum()` and `tl.minimum()`.
- Give agents a concrete default they may apply anywhere in the kernel when the argument is omitted: `propagate_nan=tl.PropagateNan.ALL`.
- Encourage optimize rounds to leave the kernel in a consistent state instead of mixing explicit and implicit NaN propagation on similar calls when that semantic change is desired.
- Warn agents that this is not a behavior-free cleanup for NaN inputs.
- Keep the workflow change small and local to skill guidance.

## Non-Goals

- Do not add static checking or runtime validation for `propagate_nan` in this change.
- Do not rewrite unrelated Triton operators or introduce a broader floating-point policy.
- Do not move this optimization-specific guidance into the CLI.

## Design

- Add a quality rule to `skills/triton-npu-optimize/SKILL.md` stating:
  - during kernel optimization work, inspect kernel compare-helper call sites such as `tl.maximum()` and `tl.minimum()` broadly rather than only the lines directly touched for the primary optimization
  - if any such kernel call does not already specify `propagate_nan`, the optimizer may add `propagate_nan=tl.PropagateNan.ALL` as a consistency repair
  - the optimizer should recognize that this can change NaN-input behavior and should not present it as a semantics-free cleanup
- Phrase the rule as explicit semantic guidance so agents can make the tradeoff intentionally instead of treating it as a mandatory style normalization.

## Expected Outcome

- Optimize rounds can make kernel compare-helper NaN propagation explicit and consistent across the operator, not only at newly touched call sites.
- Agents will be less likely to apply this rewrite blindly when the operator may rely on existing NaN behavior.
- Future prompt or skill refactors will have a design reference for why this rule exists.
