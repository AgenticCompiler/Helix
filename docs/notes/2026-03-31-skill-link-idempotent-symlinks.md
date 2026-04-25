## Summary

- Workspace skill preparation now stages copied skill directories instead of creating symlinks.
- The CLI should fail explicitly when a target workspace skill path already exists as a symlink.

## User-Visible Behavior

- Re-running the same command in a workspace with pre-existing copied skill directories should stay quiet and avoid replacing user content.
- Verbose output should report when no new skill copies were created.
- Re-running in a workspace that still contains old skill symlinks should fail with a clear message so the ambiguous layout can be fixed explicitly.

## Implementation Notes

- For Codex, if `.codex/skills` is missing, copy the repository `skills/` directory into it.
- For Codex and OpenCode, if a target per-skill path already exists as a normal directory, skip it rather than replacing it.
- If any target skill path already exists as a symlink, raise an explicit error.
- Keep cleanup limited to copied paths created during the current run.
