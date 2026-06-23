---
name: npu-analyze-commit-perf
description: Analyze Git commits on a local Ascend NPU operator branch and organize changed operators into workspace directories for downstream optimization.
---

# Git Commit Operator Organization

## Goal

Analyze the current branch relative to a base revision, identify changed operators,
and produce a workspace plan that the CLI uses to scaffold operator directories.

## Inputs

- A local Git repository or workspace path.
- A base revision, usually `origin/main`.

## Outputs

- `workspace-plan.json` — machine-readable plan consumed by `scaffold_operators.py`.
  Each entry describes one changed operator with its launch function, source path,
  and kernel dependencies.

## Required References

Read these before writing:

- [references/output-contract.md](references/output-contract.md)

When available, use the sibling `{language}-npu-optimize-knowledge` skill as the generic
optimization pattern and symptom library.

## Workflow

### Stage 1: Collect Git Context

```bash
python3 ./scripts/collect_commit_context.py \
  --base <base-revision> \
  --output .triton-agent/commit-perf-context.json
```

If `commit_count` is zero, stop immediately and explain that `<base>..HEAD` is empty.
Common causes: `HEAD` equals the base revision, or the wrong branch is checked out.

### Stage 2: Group By File

```bash
python3 ./scripts/group_commit_context_by_file.py \
  --input .triton-agent/commit-perf-context.json \
  --output .triton-agent/commit-perf-file-groups.json
```

### Stage 3: Produce Workspace Plan

Analyze the file groups to identify changed operators:

1. For each changed operator source file, identify the launch function(s) and their kernels.
2. Determine the baseline (base revision) and optimized (HEAD) versions.
3. Write `workspace-plan.json` following the contract in
   [references/output-contract.md](references/output-contract.md).

The CLI will call `scaffold_operators.py` to create operator workspace directories
from the plan.

## Hard Rules

- Keep prompts, report text, and user-visible instructions in English.
- Do not edit tracked source files.
- Do not switch branches or perform destructive Git commands.
- Do not claim benchmark speedups without evidence.
