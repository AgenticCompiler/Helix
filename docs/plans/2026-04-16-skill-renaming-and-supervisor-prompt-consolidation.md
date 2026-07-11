# Skill Renaming And Supervisor Prompt Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dedicated optimize supervisor skill, rename the remaining repository skills to a consistent `triton-npu-*` scheme, and update runtime/tests/docs to match.

**Architecture:** Keep CLI command names stable while renaming repository skill directories and all runtime references to them. Consolidate optimize supervisor behavior into the built-in supervisor prompt, then clean up documentation that still describes deleted role briefs, the removed supervisor skill, or old skill names as current behavior.

**Tech Stack:** Python 3.9+, unittest, repository markdown docs

---

### Task 1: Lock supervisor-skill removal and skill renames with failing tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_skills.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] Update prompt and runtime assertions so supervised optimize no longer expects `optimize-supervisor`.
- [ ] Update skill-staging and model tests so they expect the renamed `triton-npu-*` skill names.
- [ ] Run the focused tests and confirm they fail before implementation.

### Task 2: Remove the optimize supervisor skill and move its contract into prompts

**Files:**
- Delete: `skills/optimize-supervisor/SKILL.md`
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/optimize/execution.py`

- [ ] Expand `build_optimize_supervisor_prompt()` so it carries the audit-only rules still needed after skill deletion.
- [ ] Remove the dedicated supervisor skill reference from optimize execution.
- [ ] Delete the `skills/optimize-supervisor/` directory.

### Task 3: Rename repository skills and update code references

**Files:**
- Move: `skills/test-gen/` -> `skills/triton-npu-gen-test/`
- Move: `skills/bench-gen/` -> `skills/triton-npu-gen-bench/`
- Move: `skills/eval-gen/` -> `skills/triton-npu-gen-eval-suite/`
- Move: `skills/operator-eval/` -> `skills/triton-npu-run-eval/`
- Move: `skills/optimize/` -> `skills/triton/triton-npu-optimize/`
- Move: `skills/optimize-check/` -> `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/`
- Move: `skills/common/ascend-npu-operator-profiler/` -> `skills/triton-npu-profile-operator/`
- Move: `skills/ascend-operator-ir-analyzer/` -> `skills/triton-npu-analyze-ir/`
- Move: `skills/triton-repair-experience/` -> `skills/triton/triton-npu-repair-guide/`
- Modify: `src/helix/models.py`
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/skill_loader.py`

- [ ] Rename the repository skill directories.
- [ ] Update command-to-skill mappings and staged skill names.
- [ ] Update any hard-coded repository skill names in runtime code and tests.

### Task 4: Update skill contents and cross-skill references

**Files:**
- Modify: renamed `SKILL.md` files under `skills/`
- Modify: renamed reference docs under `skills/`

- [ ] Update each skill frontmatter `name:` to the new directory name.
- [ ] Update cross-skill references so skills point to the renamed siblings.
- [ ] Preserve existing workflow behavior and helper-script locations under the renamed directories.

### Task 5: Clean up current-facing documentation

**Files:**
- Modify: `README.md`
- Modify: selected `docs/*.md` and `docs/specs/*.md` files that describe current behavior

- [ ] Update README examples and workflow descriptions to use the new skill names where skill naming is mentioned.
- [ ] Correct current behavior descriptions around supervised optimize so they no longer claim role brief files or a dedicated supervisor skill are used today.
- [ ] Add concise supersession notes instead of rewriting historical design intent where appropriate.

### Task 6: Verify the migration

**Files:**
- Modify: none

- [ ] Run focused unit tests for prompt building, optimize runtime, skill staging, models, and generation contracts.
- [ ] Run one final search for `optimize-supervisor`, old role-brief paths, and old skill directory names in current-facing code/docs.
- [ ] Summarize any intentionally preserved historical references that remain only for archival context.
