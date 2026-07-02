# Distill Git Repo Organizer Skill Design

## User-Visible Semantics

`triton-agent distill --source git-repo` still analyzes a Git branch, writes
`.triton-agent/workspace-plan.json`, scaffolds operator workspaces, and then runs
the normal distill loop.

## Design

The dynamic Python prompt for workspace organization should only provide run-time
facts: repository root, base revision, precomputed fork revision, changed-file
extension filter, and output path. Durable instructions about how to identify
changed Triton operators and how to write `workspace-plan.json` belong in the
existing `ascend-npu-analyze-commit-perf` common skill, which already owns the
plan contract and `scaffold_operators.py`.

The git-repo organizer agent call should stage the distill skills workspace so
the agent can read `ascend-npu-analyze-commit-perf`. The CLI remains responsible
for computing the merge-base and running the scaffold script.

## Testing

Tests should verify that the generated organizer prompt routes the agent to the
staged skill and no longer embeds the full workflow, and that the git-repo agent
call passes `skills_root` for skill staging.
