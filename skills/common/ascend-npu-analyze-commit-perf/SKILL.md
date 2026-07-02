---
name: ascend-npu-analyze-commit-perf
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

When available, use the corresponding `<language>-npu-optimize-knowledge` skill
as the generic optimization pattern and symptom library. The caller prompt may
name the active operator language, such as `triton` or `tilelang`.

## Workflow

### Stage 1: Collect Git Context

If the caller prompt gives a precomputed fork revision or merge-base, use that
revision directly as the baseline for diffs and do not run `git merge-base`
again. Otherwise collect context from the requested base revision:

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

1. List changed operator source files. Prefer the caller's extension filter when
   provided, for example:
   ```bash
   git diff --name-only <fork-revision>..HEAD -- "*.py" "*.triton" "*.ttir" "*.mlir"
   ```
2. For each changed file, inspect both the diff and the full source at `HEAD`:
   ```bash
   git diff <fork-revision>..HEAD -- <source_path>
   git show HEAD:<source_path>
   ```
   Read file content silently; do not print full source to stdout.
3. Identify kernel functions in the file. For Triton, look for `@triton.jit`
   kernels and the host launch functions that call them. For TileLang, look for
   TileLang kernel definitions and their public Python entry points.
4. Only treat a kernel as changed when its body changed. Ignore import-only,
   comment-only, docstring-only, formatting-only, and unrelated helper changes.
5. For each changed kernel body, find the host-side launch function that users
   call. This is the `def` that calls the kernel via `kernel_name[grid](...)`.
   If a private helper launches the kernel, trace upward to the first public
   entry point.
6. Cross-check each candidate against the diff. If the diff does not touch the
   launch function or one of its kernels, skip it.
7. Emit one `operators[]` entry per launch function. If multiple changed kernels
   share one launch function, put all kernel names in that entry's `kernels`
   list.
8. Deduplicate by launch function and source path. If the same launch function
   appears from multiple source paths, keep one entry and explain the conflict
   in `notes`.
9. Verify each `source_path` exists at `HEAD`:
   ```bash
   git cat-file -e HEAD:<source_path> && echo ok || echo missing
   ```
   Skip entries that do not exist.
10. Write `workspace-plan.json` following the contract in
   [references/output-contract.md](references/output-contract.md).

The CLI will call `scaffold_operators.py` to create operator workspace directories
from the plan.

## Hard Rules

- Keep prompts, report text, and user-visible instructions in English.
- Do not edit tracked source files.
- Do not switch branches or perform destructive Git commands.
- Do not claim benchmark speedups without evidence.
- Do not extract or write operator `.py` files while producing the plan; the CLI
  scaffold script handles source extraction.
- Do not include every kernel in a changed file. Include only launch functions
  whose kernel bodies changed.
- Do not guess. If the diff does not clearly show a function body change, skip
  the entry.
