# `optimize-status` Single-Workspace Input Design

## Summary

- Let `optimize-status --input <dir>` inspect `<dir>` directly when that directory already contains optimize artifacts.
- Keep the existing batch-root behavior when the input directory is only a parent of multiple workspaces.

## Behavior

- Resolve `--input` as a directory as before.
- If the input directory itself contains optimize artifacts, treat it as one workspace and render only that workspace's status.
- Otherwise, keep scanning immediate child directories as batch workspaces.

## Optimize Artifact Detection

- Reuse the same signals already used by `optimize-status` status analysis:
  - `opt-note.md`
  - one or more `opt-round-*` directories
  - top-level baseline perf files such as `baseline_perf.txt` or `*_perf.txt`

## Rationale

- Users often run `optimize-status --input .` from inside one operator workspace.
- The current batch-only interpretation incorrectly scans child directories and can surface meaningless `opt-round-*` entries instead of the workspace summary the user expects.
