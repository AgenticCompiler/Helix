# Commit Performance Knowledge Report Contract

## Purpose

`PERF_KNOWLEDGE_BASE.md` records branch-level performance knowledge inferred from Git
commits. Analysis is **file-grouped and incremental**: each code file is analyzed in its
own round, then appended to the report.

## Required Structure

```markdown
# Performance Knowledge Base

## Run Summary

## Skipped Commits

## File Analyses

## Reusable Rules

## Pattern Promotion Candidates

## Limitations And Uncertainties
```

## Run Summary

Include:

- repository path
- base revision and analyzed HEAD
- target chip
- total commits in range
- hard-skipped commits
- file groups analyzed
- analysis mode: `incremental-by-file`
- when analysis was limited to selected merge requests, include:
  `| Analyzed pull requests | 99, 107 |`

## Skipped Commits

List every **hard-skipped** commit from the context JSON (helper `hard_skip: true`).
One short line per commit: SHA, subject, skip reason.

Do not list soft-filtered commits here. Commits classified as `correctness-related` or
`noise` must not appear anywhere in the report.

## File Analyses

One subsection per file path that has at least one performance-relevant commit.

Use the format in [incremental-file-analysis.md](incremental-file-analysis.md).

Each file section includes **only** commits classified as `performance-related` or
`rollback-or-negative`, in chronological order.

Files with no performance-relevant commits after soft classification are omitted
entirely from this section.

## Reusable Rules

Written in the final synthesis round. Deduplicate lessons from performance-relevant file
sections only. Do not invent rules from omitted or non-performance commits.

## Pattern Promotion Candidates

Written in the final synthesis round.

## Limitations And Uncertainties

Written in the in-report synthesis round. Include files or commits where evidence was weak.

## Final Consolidated Report

The incremental report is not the only deliverable. After file rounds complete, also
write `PERF_PATTERN_SYNTHESIS.md` following
[pattern-synthesis-contract.md](pattern-synthesis-contract.md).

That file is the user-facing summary for pattern clustering, pattern-index alignment, and
skill-update recommendations.
