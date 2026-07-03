# Skill Logical Cross-Reference Wording Design

## Background

Live skill documents still contain some cross-skill references that encode grouped source-tree paths such as `../other-skill/SKILL.md`, `../other-skill/references/...`, or `../other-skill/scripts/...`.

That wording is brittle now that the repository owns a logical-skill catalog and stages skills into a flat backend-visible layout. The user-facing contract should describe inter-skill handoffs by logical skill name, not by repository-relative source paths.

## Decision

Use logical skill names for every cross-skill workflow handoff in live skill documentation.

- When a workflow needs another skill, say `use the skill \`<logical-skill-name>\``.
- When a workflow needs a resource from another skill, name it as that skill's `references/...` file instead of linking to `../...`.
- When a workflow depends on another skill's helper command, describe it as that skill's helper or subcommand instead of embedding a sibling `../.../scripts/...` path.
- Keep repository-relative script paths only for helpers that live inside the same logical skill directory as the current document.

## Scope

Update the live skill docs that still expose cross-skill relative paths:

- `ascend-npu-analyze-ir`
- `ascend-npu-analyze-round-performance`
- `ascend-npu-gen-eval-suite`
- `ascend-npu-profile-operator`
- `ascend-npu-kernel-bench-logs`
- `triton-npu-optimize`
- `triton-npu-optimize/references/artifacts.md`

Also update contract tests so they assert the logical-skill wording rather than the old relative-path wording.

## Non-goals

- No skill staging behavior changes
- No helper-script behavior changes
- No pattern-index generation changes

## Verification

Run the generation-contract test coverage that reads these live skill documents and confirms the new wording contract.
