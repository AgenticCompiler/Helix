# Skill Staging Rule Table Design

## Summary

- Centralize command-to-skill staging decisions in one shared table.
- Use a tiny directive syntax so each command's staging intent stays readable:
  - `*` means stage all repository skills.
  - `+skill-name` means include that skill.
  - `-skill-name` means remove that skill from the current selection.
- Keep backend copy behavior unchanged; this only changes how each subcommand chooses which skills to stage.

## Problem

- `gen-eval` currently carries its own staging list in generation orchestration.
- `convert` and `optimize` define their own staged skill sets in separate modules.
- That makes staging behavior easy to drift and hard to audit.

## Proposed Shape

- Add one shared module for staging rules and resolution.
- Store command rules in a single table keyed by `CommandKind`.
- Resolve each rule into:
  - `staged_skill_names`
  - optional `staged_skill_sources`
- Keep aliasing for special cases like optimize knowledge versioning separate from the basic include/exclude syntax.

## Non-Goals

- Do not change `SkillLinkManager` copy semantics.
- Do not change prompt text, output paths, or backend launch behavior.
- Do not add new CLI flags.
