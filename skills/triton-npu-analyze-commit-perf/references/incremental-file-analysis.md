# Incremental File-Grouped Analysis

## Why File Groups

Analyze one code file per round instead of the whole branch at once. Each round looks at
every commit that touched the same file, in chronological order, so the agent can follow
how optimizations evolved on that hot path.

## Required Artifacts

- `.triton-agent/commit-perf-context.json` from `collect_commit_context.py`
- `.triton-agent/commit-perf-file-groups.json` from `group_commit_context_by_file.py`
- `.triton-agent/commit-perf-analysis-state.json` for resume progress
- `PERF_KNOWLEDGE_BASE.md` (or the requested output path)

## Round Flow

1. Run `collect_commit_context.py` for `<base>..HEAD`.
2. Run `group_commit_context_by_file.py` on the context JSON.
3. If `commit_count` is zero, stop with an actionable error. Do not write a fake report.
4. Initialize the report skeleton once:
   - `# Performance Knowledge Base`
   - `## Run Summary`
   - `## Skipped Commits`
   - `## File Analyses` (empty placeholder is fine)
   - `## Reusable Rules` (placeholder: pending synthesis)
   - `## Pattern Promotion Candidates` (placeholder)
   - `## Limitations And Uncertainties` (placeholder)
5. Write `.triton-agent/commit-perf-analysis-state.json` with:
   - `completed_files`: `[]`
   - `pending_files`: all paths from `file_groups`
6. For each file path in `pending_files`, one round at a time:
   - Read only that file's `commits` array from `commit-perf-file-groups.json`.
   - Read each commit's `subject`, `body`, and full `message` before soft classification.
   - Soft-classify each commit before writing anything.
   - **Do not write performance-unrelated commits into the report.**
   - Append a file section only when the file has at least one
     `performance-related` or `rollback-or-negative` commit worth documenting.
   - Update `completed_files` and remove the path from `pending_files` even when the
     file is omitted from the report (record omission in the state file).
   - Save state after each file round so a resumed run can continue.
7. After all file groups are done, run a short in-report synthesis round on
   `PERF_KNOWLEDGE_BASE.md`:
   - Fill `## Reusable Rules`
   - Fill `## Pattern Promotion Candidates`
   - Fill `## Limitations And Uncertainties`
   - Update `## Run Summary` with final counts.
8. Run the final pattern synthesis round described in
   [pattern-synthesis-contract.md](pattern-synthesis-contract.md):
   - Read the completed `PERF_KNOWLEDGE_BASE.md`
   - Compare against staged `triton-npu-optimize-knowledge/references/pattern_index.md`
   - Write the consolidated report (default `PERF_PATTERN_SYNTHESIS.md`)

## Per-File Section Format

```markdown
### src/path/to/kernel.py

- Commits analyzed: <count>
- Commit SHAs: <short-sha list in chronological order>
- Overall theme:

#### Commit Timeline

Include **only** commits classified as `performance-related` or `rollback-or-negative`.

Do **not** create timeline entries for `correctness-related`, `noise`, or commits that
are only formatting, docs, tests, CI, imports, or comment-only changes.

##### <short-sha> <subject>
- Commit message notes: (summarize relevant body text when it adds evidence)
- Classification:
- Confidence:
- What changed:
- Hardware mechanism:
- Reusable rule:

#### Cross-Commit Lessons For This File

#### Pattern Links

#### IR Or Compiler Evidence
```

## Resume Rules

- If `commit-perf-analysis-state.json` exists and lists pending files, skip files already
  in `completed_files`.
- Append new file sections only. Do not rewrite earlier completed file sections unless the
  user explicitly requested a full regeneration with `--force`.
- Keep the report append-only during file rounds.

## Performance-Only Output Rules

Write to `PERF_KNOWLEDGE_BASE.md` only when the content is performance knowledge.

| Classification | Write to report? |
| --- | --- |
| `performance-related` | Yes, full timeline entry |
| `rollback-or-negative` | Yes, full timeline entry (failed optimization lesson) |
| `uncertain` | Only if static evidence suggests a real perf mechanism; otherwise omit |
| `correctness-related` | **No** |
| `noise` | **No** |

`## Skipped Commits` is only for helper hard-skipped commits (one short line each).
Do not copy soft-filtered commits into `## Skipped Commits` or `## File Analyses`.

If a file has no performance-relevant commits after soft classification, **do not**
append a `### <file-path>` section. Record in state:

```json
"omitted_files": [
  {"path": "src/...", "reason": "no performance-relevant commits after soft classification"}
]
```

## Hard Rules

- Never analyze more than one file group in a single reasoning round before writing.
- Never skip a file group silently without updating the state file.
- Never dump non-performance commit narratives into the report.
