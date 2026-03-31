## Summary

- Skill link preparation should treat an existing symlink to the current repository skills as already satisfied.
- The CLI should skip recreating such symlinks instead of failing or reporting them as new links.

## User-Visible Behavior

- Re-running the same command in a workspace that already links to this repository's skills should be quiet and idempotent.
- Verbose output should report that no new links were created when the workspace is already wired correctly.

## Implementation Notes

- For Codex, check whether `.codex/skills` already exists as a symlink to the repository `skills/` directory before trying to create it.
- For both Codex per-skill links and OpenCode per-skill links, continue skipping existing symlinks that already point at the expected source skill directory.
- Keep cleanup limited to links created during the current run.
