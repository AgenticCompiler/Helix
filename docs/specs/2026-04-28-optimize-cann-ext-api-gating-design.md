# Optimize CANN Extension API Gating Design

## Goal

Add an `optimize` option that explicitly allows A5-only CANN Triton extension API optimization patterns, without exposing those patterns in other optimize runs.

## User-visible behavior

- Add `--enable-cann-ext-api` to `optimize` and `optimize-batch`.
- Default behavior stays unchanged: optimize runs do not expose the new CANN extension API patterns.
- When `--enable-cann-ext-api` is set with `--target-chip A5`, the optimize agent may read and use a dedicated staged skill that contains the specialized patterns.
- When `--enable-cann-ext-api` is set with any non-`A5` target chip, the CLI must fail fast with a clear validation error.

## Design

### Access control boundary

Access control lives in optimize orchestration, not in backend-specific policy code.

- CLI parsing records the explicit user opt-in.
- Optimize validation enforces the `A5`-only rule.
- Optimize request construction decides whether the dedicated skill is staged.
- Optimize prompt text decides whether the agent is told that the capability is available.

This keeps the capability unavailable by default and avoids relying on the agent to self-restrict after the fact.

### Skill shape

Add a dedicated skill directory for the new material instead of mixing it into the default `triton-npu-optimize` skill.

- New skill: `triton-npu-cann-ext-api-patterns`
- Purpose: hold the specialized A5-only CANN Triton extension API pattern references
- Benefit: the existing optimize skill remains the default contract, while the specialized material is only copied into staged workspace skills when explicitly authorized

### Prompt semantics

When the feature is enabled, optimize prompts should add a short explicit allowance:

- the run has CANN extension API pattern access enabled
- the dedicated staged skill is the source of truth for those patterns
- those patterns are A5-specific guidance and should not be treated as generic optimize defaults

When the feature is disabled, optimize prompts must not mention this capability.

## Implementation notes

- Extend `OptimizeRunOptions` with `enable_cann_ext_api: bool`.
- Extend optimize option validation with the new target-chip constraint.
- Extend optimize request building so `staged_skill_names` conditionally appends the new skill.
- Thread the enable flag into optimize prompt builders so the prompt can mention the staged capability only when enabled.
- Update README optimize docs to describe the new option and its A5-only requirement.
- Add tests for CLI parsing, validation failure, conditional prompt text, and conditional skill staging.

## Non-goals

- No backend-specific permission system changes
- No automatic hardware detection
- No implicit enablement based on chip target alone
