# Skill Renaming And Supervisor Prompt Consolidation Design

## Summary

This change simplifies optimize supervision and makes repository skill names consistent. The dedicated `optimize-supervisor` skill will be removed, its audit-only behavior will move into the built-in supervisor prompt, and the remaining repository skills will be renamed to a uniform function-oriented `triton-npu-*` scheme.

## Goals

- Remove the redundant `optimize-supervisor` skill.
- Keep supervisor behavior explicit through the runtime prompt instead of a second skill contract.
- Rename the remaining repository skills to a consistent function-oriented pattern.
- Preserve existing CLI commands and optimize orchestration behavior.
- Clean up current-facing documentation that still describes deleted role briefs, removed supervisor skills, or old skill names.

## Non-Goals

- Do not redesign the optimize loop or change gate semantics.
- Do not move workflow logic out of repository skills into new CLI subcommands.
- Do not rename CLI commands such as `gen-test`, `optimize`, or `run-bench`.
- Do not rewrite historical plans and specs as if they never existed.

## Naming Scheme

Use `triton-npu-<verb>-<target>` for repository skills.

Rename the remaining skills as follows:

- `test-gen` -> `triton-npu-gen-test`
- `bench-gen` -> `triton-npu-gen-bench`
- `eval-gen` -> `triton-npu-gen-eval-suite`
- `operator-eval` -> `triton-npu-run-eval`
- `optimize` -> `triton-npu-optimize`
- `optimize-check` -> `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`
- `ascend-npu-operator-profiler` -> `triton-npu-profile-operator`
- `ascend-operator-ir-analyzer` -> `triton-npu-analyze-ir`
- `triton-repair-experience` -> `triton-npu-repair-guide`

## Supervisor Prompt Consolidation

The optimize supervisor role currently gets behavior from both:

- the built-in `build_optimize_supervisor_prompt()` prompt text
- `skills/optimize-supervisor/SKILL.md`

That duplication makes the supervisor workflow harder to reason about and conflicts with the current implementation direction, where role behavior already comes from launch prompts plus `.helix/round-brief.md` and `.helix/supervisor-report.md`.

After this change:

- `skills/optimize-supervisor/` is deleted
- supervised optimize uses the renamed optimize skill staging plus the built-in supervisor prompt
- `build_optimize_supervisor_prompt()` becomes the only supervisor-role contract
- the prompt must carry any still-needed audit restrictions that previously lived only in the deleted skill

## Runtime And Test Impact

The implementation must update:

- `COMMAND_TO_SKILL`
- optimize execution code that currently hard-codes `optimize-supervisor`
- generation staged-skill names
- test fixtures that stage, assert, or copy repository skill directories
- prompt assertions that mention old skill names

## Documentation Cleanup Boundaries

Update current-facing docs to describe the new skill names and the current supervised optimize behavior.

Keep historical design documents, but add small corrective notes where they would otherwise mislead readers about current runtime behavior, especially around:

- `.helix/roles/*`
- `optimize-supervisor`
- old skill directory names referenced as current behavior

## Verification

- Update targeted tests first so skill removal and renames fail before implementation.
- Run focused unit tests for prompt building, optimize runtime, skill staging, and generation contracts.
- Run a targeted repo-wide search after implementation to catch stale references in current-facing docs and code.
