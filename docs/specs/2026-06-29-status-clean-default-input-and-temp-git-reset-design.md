# Status/Clean Default Input And Temporary Git Reset Design

## Summary

Adjust two CLI defaults and one cleanup rule:

- `status` and `clean` should operate on the current directory when `--input` is omitted.
- Temporary `.git` repositories created by skill staging should be preserved by default.
- Cleanup should remove that temporary `.git` only when `HELIX_RESET_GIT_REPO` is set.

## Goals

- Let users run `helix status` and `helix clean` directly inside a workspace.
- Avoid deleting a temporary workspace-local git repo unless the caller explicitly requests reset behavior.
- Keep the change local to CLI argument defaults and skill-staging cleanup semantics.

## Non-Goals

- Do not change input semantics for other commands.
- Do not delete any pre-existing user-owned `.git` directory or gitfile.
- Do not broaden cleanup to inspect parent repositories or arbitrary git state.

## User-Visible Behavior

- `uv run helix status` behaves like `uv run helix status --input .`
- `uv run helix clean` behaves like `uv run helix clean --input .`
- When skill staging creates `workdir/.git` for the current run, later cleanup keeps it by default.
- If `HELIX_RESET_GIT_REPO` is set to a truthy value, cleanup removes only the temporary `.git` created by that run.

## Implementation Notes

- Change the parser so `status` and `clean` use `.` as the default `--input` value instead of requiring the flag.
- Keep command handlers unchanged apart from consuming the parser-provided default.
- Gate `SkillLinkManager.cleanup()` temporary-git removal on both:
  - `link_set.temporary_git_dir is not None`
  - `HELIX_RESET_GIT_REPO` being enabled
- Preserve rollback-on-failure behavior during `prepare_skills()`: if staging fails after creating a temporary `.git`, remove it before re-raising.

## Testing

- Parser coverage proving `status` and `clean` accept omitted `--input` and default to `.`
- Handler coverage proving omitted `--input` resolves against the current working directory
- Skill cleanup coverage proving temporary `.git` is preserved by default
- Skill cleanup coverage proving temporary `.git` is removed when `HELIX_RESET_GIT_REPO` is enabled
