# Distill Command Refactor Design

## User-Visible Semantics

`triton-agent distill -i <input>` replaces the current `diff-skills-update`
command. The command distills optimization evidence into reusable optimize
knowledge pattern cards, then verifies the distilled guidance by asking an agent
to regenerate the optimization from the baseline and staged skills alone.

The public command name is `distill`. The Python package and command module use
the same name so the codebase no longer carries the old implementation-centric
`diff_skills_update` name. The old `diff-skills-update` command is removed
rather than kept as a compatibility alias.

## Architecture

The CLI remains a thin orchestrator. It discovers operator pairs or optimize
workspaces, prepares the editable optimize-knowledge workspace, stages a
dedicated common workflow skill, and launches agents for three roles:

- distill evidence into pattern updates;
- simulate an optimizer using only the baseline and staged skills;
- analyze mismatches and refine the skills.

Large prompt bodies move out of Python into
`skills/common/ascend-npu-distill-patterns`. Python prompt builders keep only
task-specific paths, output JSON contracts, and compact role dispatch text. The
new skill owns durable natural-language workflow guidance for reading optimize
evidence, semantic pattern matching, card authoring, and simulation/analysis
boundaries.

## Skill Changes

`ascend-npu-distill-patterns` becomes the staged workflow skill for the distill
command. It incorporates the reusable parts of
`ascend-npu-kernel-bench-logs`: read `opt-note.md` first, use round artifacts as
evidence, map mechanisms semantically instead of by citation, create a new card
only when no existing `## Summary` / `## Use When` fits, and keep pattern cards
generic and self-contained.

NPUKernelBench-specific progress tables, field inventories, narrative ledgers,
and manual synthesis bookkeeping are intentionally not carried forward. The old
`ascend-npu-kernel-bench-logs` skill is removed from the repository catalog and
deleted.

## Data Flow

For `code-diff`, the command still scans operator directories for `opt_*.py`
paired with the baseline file of the same name without the `opt_` prefix.

For `optimize-process`, the command still reads optimize workspaces using
`baseline/state.json`, `opt-note.md`, `learned_lessons.md`, and round summaries
to identify the baseline and final optimized operator.

For `git-repo`, the organizer still creates temporary operator workspaces from a
git merge-base plan before the same distillation loop runs.

Each pair writes reports under `simulate/`, including matched patterns, updated
patterns, iteration results, and final status.

## Testing

Focused tests cover the new command name, command kind, configuration defaults,
staged skills, renamed package imports, prompt references to the staged skill,
and removal of the old kernel-bench-logs catalog entry. Existing discovery,
workflow, and skills workspace behavior remains covered after import/name
updates.
