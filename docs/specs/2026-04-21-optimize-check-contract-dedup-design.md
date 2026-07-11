# Optimize Check Contract Dedup Design

## Summary

- Remove duplicated optimize-check models and contract parsing logic between `src/helix/optimize/` and `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/`.
- Keep `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` as the source of truth for optimize validation contracts.
- Preserve direct `python3 scripts/optimize_check.py ...` execution inside staged skill copies.

## Goals

- Deduplicate `OptimizeCheckResult`, `BaselineState`, and `RoundState`.
- Deduplicate baseline and round artifact inspection plus state-loading helpers.
- Keep runtime imports and public helper functions stable for existing callers.
- Keep the skill script runnable without depending on the repository `src/` tree in staged workspaces.

## Non-Goals

- Do not change optimize baseline or round validation semantics.
- Do not move optimize worker workflow rules out of the skill.
- Do not change the public `optimize_check.py` CLI shape or exit-code behavior.

## Design

### Skill-Owned Shared Contract Module

- Add a new shared helper module under `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/`.
- Move these items into that helper:
  - `OptimizeCheckResult`
  - `BaselineState`
  - `BaselineArtifactsInspection`
  - `RoundState`
  - `RoundArtifactsInspection`
  - baseline and round JSON loading helpers
  - baseline and round artifact inspection helpers
  - baseline gate and round check evaluation helpers

This keeps the reusable validation contract inside the optimize-check skill, which matches the existing design that treats the skill as the technical source of truth.

### Thin Skill CLI Wrapper

- Reduce `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check.py` to a thin CLI wrapper that imports shared types and check functions from the new helper.
- Keep the wrapper self-contained within the skill directory so direct execution still works after skills are copied into a workspace.

### Runtime Bridge Layer

- Add a small runtime bridge module under `src/helix/optimize/` that loads the shared skill helper through the existing skill-loader path.
- Re-export the shared dataclasses from `src/helix/optimize/models.py`.
- Re-export the shared baseline and round contract functions from `src/helix/optimize/baseline.py` and `src/helix/optimize/round_contract.py`.

This keeps existing runtime call sites stable while removing duplicated implementations.

## Testing

- Add a regression test that the optimize-check skill module and runtime models share the same dataclass identities.
- Add baseline and round tests that compare runtime wrapper results with the shared skill helper results.
- Keep the existing direct-script help test for `optimize_check.py`.

## Expected Outcome

- Optimize contract models and parsing logic live in one place.
- Runtime callers continue to use the existing `src/helix/optimize/*` APIs.
- Staged skills remain directly executable in copied workspaces.
