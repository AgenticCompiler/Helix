# ArgSort AiCPU Fallback Pattern Design

## Goal

Capture the Ascend-specific `torch.argsort()` integer-dtype fallback behavior in the optimize knowledge base so future diagnosis can route directly to the relevant workaround.

## Decision

Add a new pattern card named `argsort-avoid-aicpu-fallback` and a new symptom card named `unsupported-dtype-fallback`.

## Why A New Pattern

This case does not fit the existing pattern cards cleanly:

- It is not primarily an algebraic rewrite. The main win comes from avoiding an unsupported backend path, not from reducing passes or changing mathematical structure.
- It is not the same as `vec-cmp`, which focuses on Triton-kernel integer compare lowering. This case is a framework-level `torch.argsort()` capability gap that triggers an AiCPU fallback.
- The runtime warning and profiler signature are central evidence, so future routing should be able to discover this behavior directly.

## Pattern Scope

The pattern should teach this narrow rule:

- On Ascend, `ArgSort` on `int32` or `int64` may fall back to AiCPU.
- When the integer key domain is exactly representable in `float32`, casting keys to `float32` before `torch.argsort()` can keep the sort on AiCore.
- This is only safe when semantic ordering is preserved under the cast.

The card must stay specific to `ArgSort` rather than generalizing to `topk`, arbitrary selection ops, or every integer-key rewrite.

## Symptom Scope

The new symptom card should route cases where:

- runtime logs explicitly report unsupported dtype or AiCPU fallback
- profiler latency is anomalously high for a tiny operator because dispatch and fallback overhead dominate
- the active framework path uses a dtype/backend combination that may have an equivalent supported alternative

## Required Documentation Changes

- Add `skills/triton/triton-npu-optimize-knowledge/references/patterns/argsort-avoid-aicpu-fallback.md`
- Add `skills/triton/triton-npu-optimize-knowledge/references/symptoms/unsupported-dtype-fallback.md`
- Regenerate the checked-in pattern and symptom indexes

## Non-Goals

- Do not widen the rule to claim that all sort-like operators share the same fallback behavior.
- Do not change existing generic scalar or transfer symptoms to own this case.
- Do not encode private benchmark-case labels as the public pattern contract.
