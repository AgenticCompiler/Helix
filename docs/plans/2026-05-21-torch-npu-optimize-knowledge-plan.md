# Torch NPU Optimize Knowledge Implementation Plan

**Goal:** Split Torch NPU operator-level optimize knowledge into a dedicated `torch-npu-optimize-knowledge` skill that is staged only for operator-target optimize runs.

**Architecture:** Keep `triton-npu-optimize-knowledge` as the generic Triton/kernel knowledge pack and add one operator-only Torch NPU knowledge pack with its own pattern index. Thread `optimize_target` into skill staging so prompts and staged references match the requested optimization scope.

**Tech Stack:** Python, unittest, Markdown skill references, generated checked-in pattern indexes

---

## Task 1: Lock the New Staging Contract With Tests

**Files:**
- Modify: `tests/test_skill_staging.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`

- Add tests that assert `resolve_staged_skills(..., optimize_target="kernel")` does not stage `torch-npu-optimize-knowledge`.
- Add tests that assert `resolve_staged_skills(..., optimize_target="operator")` does stage `torch-npu-optimize-knowledge`.
- Add optimize-request tests that assert `build_optimize_request(... optimize_target="operator")` includes the new skill in `staged_skill_names`.
- Add prompt tests that assert operator-target optimize guidance mentions the Torch NPU knowledge skill while kernel-target guidance does not.

## Task 2: Implement Target-Aware Skill Staging

**Files:**
- Modify: `src/triton_agent/skill_staging.py`
- Modify: `src/triton_agent/optimize/orchestration.py`

- Extend `resolve_staged_skills()` to accept `optimize_target`.
- Keep the existing optimize-knowledge source remapping behavior for `triton-npu-optimize-knowledge`.
- Append `torch-npu-optimize-knowledge` only when `command_kind == OPTIMIZE` and `optimize_target == "operator"`.
- Thread `options.optimize_target` from optimize orchestration into skill staging.

## Task 3: Add the New Torch NPU Knowledge Skill And Migrate References

**Files:**
- Create: `skills/torch-npu-optimize-knowledge/SKILL.md`
- Create: `skills/torch-npu-optimize-knowledge/references/patterns/argsort-avoid-aicpu-fallback.md`
- Create: `skills/torch-npu-optimize-knowledge/references/pattern_index.md`
- Delete: `skills/triton-npu-optimize-knowledge/references/patterns/argsort-avoid-aicpu-fallback.md`
- Modify: `skills/triton-npu-optimize-knowledge/references/pattern_index.md`

- Keep the detailed pattern content intact except for ownership wording if needed.
- Regenerate the checked-in pattern index for the Torch NPU knowledge skill from the shared builder script.

## Task 4: Update Optimize Guidance To Reference The New Skill Correctly

**Files:**
- Modify: `src/triton_agent/optimize/prompts.py`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Modify: `tests/test_generation_contracts.py`

- Make the core layered-analysis guidance mention the generic Triton knowledge skill in both targets.
- Add operator-target-only guidance that points to `torch-npu-optimize-knowledge` for Torch NPU and whole-operator patterns.
- Update skill contract tests to reflect the new ownership split.

## Task 5: Verify Index Generators, Skill Docs, And Runtime Behavior

**Files:**
- Modify: `tests/test_optimize_pattern_tools.py`

- Add generator and checked-in index tests for the new Torch NPU knowledge skill.
- Re-run the generic knowledge generator checks after migration.
- Run focused unit tests first, then repository-standard verification if the targeted checks pass.
