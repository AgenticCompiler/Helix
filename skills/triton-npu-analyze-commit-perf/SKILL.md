---
name: triton-npu-analyze-commit-perf
description: Analyze Git commits on a local Triton Ascend NPU operator branch, infer performance mechanisms from diffs and context, and write PERF_KNOWLEDGE_BASE.md incrementally by file.
---

# Git Commit Performance Analysis

## Goal

Analyze the current branch relative to a base revision and produce a reusable performance
knowledge document. This is a post-hoc Git history analysis workflow, not a live
optimization round.

Use incremental **file-grouped** analysis: one code file per round, all commits that
touched that file analyzed together, results appended to the report after each round.

## Inputs

- A local Git repository or workspace path.
- A base revision, usually `origin/main`.
- A target chip, usually `A5`.
- A requested incremental report path, usually `PERF_KNOWLEDGE_BASE.md`.
- A requested final synthesis report path, usually `PERF_PATTERN_SYNTHESIS.md`.
- Optional merge-request filter (`--pull-request` / `--pr`) to analyze only commits from
  selected GitCode/GitLab merge requests (`!N`).

## Outputs

- `PERF_KNOWLEDGE_BASE.md` or the requested incremental report (built file-by-file).
- `PERF_PATTERN_SYNTHESIS.md` or the requested synthesis report (written once at the end).
- `.triton-agent/commit-perf-context.json`
- `.triton-agent/commit-perf-file-groups.json`
- `.triton-agent/commit-perf-analysis-state.json` for resume tracking

## Required References

Read these before writing:

- [references/output-contract.md](references/output-contract.md)
- [references/incremental-file-analysis.md](references/incremental-file-analysis.md)
- [references/pattern-synthesis-contract.md](references/pattern-synthesis-contract.md)

When available, use the sibling `triton-npu-optimize-knowledge` skill as the generic
optimization pattern and symptom library.

## Workflow

### Stage 1: Collect Git Context

```bash
python3 .codex/skills/triton-npu-analyze-commit-perf/scripts/collect_commit_context.py \
  --base <base-revision> \
  --output .triton-agent/commit-perf-context.json \
  [--pull-request <N> ...]
```

When the CLI passes `--pull-request`, repeat the same flags here. Record the selected PR
IIDs in `## Run Summary` as `| Analyzed pull requests | ... |`.

If `commit_count` is zero, stop immediately and explain that `<base>..HEAD` is empty.
Common causes: `HEAD` equals the base revision, or the wrong branch is checked out.

### Stage 2: Group By File

```bash
python3 .codex/skills/triton-npu-analyze-commit-perf/scripts/group_commit_context_by_file.py \
  --input .triton-agent/commit-perf-context.json \
  --output .triton-agent/commit-perf-file-groups.json
```

Each file group contains every non-skipped commit that touched that file, in
chronological order, with per-file diffs.

### Stage 3: Initialize Report And State

Follow [references/incremental-file-analysis.md](references/incremental-file-analysis.md).

Create the report skeleton and state file once. Record all hard-skipped commits in
`## Skipped Commits` during initialization.

### Stage 4: Analyze One File Per Round

For each entry in `file_groups`:

1. Read only that file's commits from `commit-perf-file-groups.json`.
2. Read each commit's `subject`, `body`, and full `message` before soft classification.
   Important performance rationale often appears only in the body.
3. Soft-classify each commit (`performance-related`, `rollback-or-negative`,
   `correctness-related`, `noise`, `uncertain`).
4. Write **only** `performance-related` and `rollback-or-negative` commits into the
   report. Do not output `correctness-related`, `noise`, or other non-performance
   commits to `PERF_KNOWLEDGE_BASE.md`.
5. Explain the evolution across the performance-relevant commits on this file.
6. Append a file section only when at least one commit deserves performance analysis.
7. Update `commit-perf-analysis-state.json` before starting the next file.

Do not batch multiple files into one analysis round.

### Stage 5: In-Report Synthesis

After all file groups are complete, update `PERF_KNOWLEDGE_BASE.md`:

- `## Reusable Rules`
- `## Pattern Promotion Candidates`
- `## Limitations And Uncertainties`
- final counts in `## Run Summary`

### Stage 6: Pattern Synthesis And Skill Alignment

Mandatory final round. Follow
[pattern-synthesis-contract.md](references/pattern-synthesis-contract.md).

1. Read the completed incremental report.
2. Cluster similar performance lessons into consolidated pattern groups.
3. List every item with source commits, mechanism, and reusable rule.
4. Compare each group or item against
   `triton-npu-optimize-knowledge/references/pattern_index.md`.
5. State relationship: `matches`, `partial-overlap`, `extends`, `novel`, or
   `contradicts`.
6. Recommend skill updates per item:
   `no-change`, `extend-existing-card`, `promote-new-pattern-card`, `local-only`,
   or `reject`.
7. Write the consolidated report to `PERF_PATTERN_SYNTHESIS.md` (or the requested
   synthesis output path).

Do not edit pattern cards or the generated index in this workflow.

### Stage 7: Optional IR Evidence

Use IR evidence only when enabled and artifacts are already available. Do not invent
compiler details.

## Hard Rules

- Keep prompts, report text, and user-visible instructions in English.
- Do not edit tracked source files.
- Do not switch branches or perform destructive Git commands.
- Append file analyses incrementally; do not wait until the end to write everything.
- Do not silently drop pending file groups; record omitted non-performance files in state.
- Do not write performance-unrelated commit analysis into the report.
- Do not claim benchmark speedups without evidence.
