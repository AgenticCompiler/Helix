# Torch NPU Optimize Knowledge Split

## Summary

- Add a new reference-only skill named `torch-npu-optimize-knowledge`.
- Keep `triton-npu-optimize-knowledge` focused on Triton kernel-oriented generic optimize patterns.
- Stage `torch-npu-optimize-knowledge` only for `--optimize-target operator`.
- Move the current Torch NPU operator-level `argsort-avoid-aicpu-fallback` pattern out of the generic Triton knowledge skill into the new Torch NPU knowledge skill.

## Problem

The current generic optimize knowledge skill mixes two different knowledge scopes:

- Triton kernel optimization guidance
- Torch NPU or whole-operator optimization guidance

The `argsort-avoid-aicpu-fallback` pattern is a concrete example. It describes a `torch.argsort()` dtype fallback rewrite that addresses Torch NPU operator placement and runtime fallback behavior, not Triton kernel implementation strategy.

That mismatch causes two problems:

- kernel-target optimize runs see operator-level advice that is outside their intended optimization scope
- the generic Triton knowledge library no longer clearly means "kernel-oriented generic optimize knowledge"

## Goals

- Keep the generic Triton optimize knowledge library semantically clean.
- Make Torch NPU and operator-level optimization knowledge an explicit opt-in for operator-target optimize runs.
- Reuse the existing staged-skill model instead of inventing a second optimize command.
- Preserve pattern-index routing for both knowledge packs.

## Non-Goals

- Do not redesign the optimize target CLI surface.
- Do not change `compare-perf` behavior in this change.
- Do not introduce automatic runtime fallback from kernel mode into operator mode.
- Do not migrate unrelated generic Triton patterns into the new skill in this iteration.

## Design

### New Skill

Add a new skill directory:

- `skills/torch-npu-optimize-knowledge/`

This skill is reference-only, like `triton-npu-optimize-knowledge`. It owns its own:

- `references/patterns/`
- `references/pattern_index.md`

Its purpose is Torch NPU and whole-operator optimization knowledge that is broader than Triton kernel-only guidance.

### Skill Boundary

After this change:

- `triton-npu-optimize-knowledge` remains the generic Triton/kernel-oriented library
- `torch-npu-optimize-knowledge` becomes the Torch NPU / operator-level pattern library

The initial Torch NPU skill content is:

- pattern: `argsort-avoid-aicpu-fallback`

The generic skill should no longer reference those items in its checked-in indexes.

### Target-Aware Staging

Optimize staging should become target-aware:

- `optimize_target=kernel`
  - stage `triton-npu-optimize-knowledge`
  - do not stage `torch-npu-optimize-knowledge`
- `optimize_target=operator`
  - stage `triton-npu-optimize-knowledge`
  - also stage `torch-npu-optimize-knowledge`

This preserves generic Triton knowledge in both modes while making operator-level knowledge explicit only when the user requested operator optimization.

### Prompt And Workflow Guidance

Optimize prompts and skill docs should teach the agent that:

- generic pattern triage starts from `triton-npu-optimize-knowledge`
- operator-target runs may additionally use `torch-npu-optimize-knowledge` for Torch NPU, framework-op fallback, data-movement, scheduling, and wrapper-level optimization directions

Kernel-target prompts should not mention the Torch NPU knowledge skill.

### Routing Rules

Pattern routing should stay local to the knowledge pack that owns the detailed references.

That means:

- `argsort-avoid-aicpu-fallback` should be removed from the generic pattern index
- the new Torch NPU skill should own the checked-in generated pattern index that points to that reference

## Alternatives Considered

### 1. Keep the pattern in the generic skill and rely on prompt wording

Pros:

- smallest code change

Cons:

- keeps the knowledge boundary muddy
- kernel-target runs still see operator-level routing material

### 2. Tag the pattern as operator-only inside the generic skill

Pros:

- avoids a new skill directory

Cons:

- still mixes two semantically different knowledge trees
- makes index triage noisier

### 3. Split into a dedicated Torch NPU knowledge skill

Pros:

- clean knowledge ownership
- target-aware staging is explicit
- matches the repository preference for specialized packs to stay separate unless explicitly needed

Cons:

- requires new staging, prompt, and test coverage

## Recommendation

Use alternative 3.

The optimize target feature already creates the right product boundary. A dedicated `torch-npu-optimize-knowledge` skill keeps kernel-oriented and operator-oriented knowledge separate while preserving the current optimize workflow structure.
