# Distill Git Repo Workspace Plan Design

## User-Visible Semantics

`helix distill --source git` still analyzes a Git branch, writes
`.helix/workspace-plan.json`, scaffolds operator workspaces, and then runs
the normal distill loop.

## Design

The dynamic Python prompt for git-repo workspace planning should only provide
run-time facts: repository root, active operator language, base revision,
precomputed fork revision, changed-file extension filter, and output path.
Durable instructions about how to identify changed operators and how to write
`workspace-plan.json` belong in the
`ascend-npu-plan-git-operator-workspaces` common skill, which owns the plan
contract and `scaffold_operators.py`.

The git-repo workspace-plan agent call should stage the distill skills workspace
so the agent can read `ascend-npu-plan-git-operator-workspaces`. The CLI remains
responsible for computing the merge-base and running the scaffold script.

Intermediate `.helix` cleanup must only remove a real directory owned by
the input workspace. If `.helix` is a symbolic link, cleanup should skip
it and leave the link target untouched.

## Testing

Tests should verify that the generated workspace-plan prompt routes the agent to
the staged skill and no longer embeds the full workflow, includes the active
operator language, that the git-repo agent call passes `skills_root` for skill
staging, and that intermediate cleanup skips `.helix` symbolic links.
