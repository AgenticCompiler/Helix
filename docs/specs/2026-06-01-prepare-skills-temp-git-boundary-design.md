# Prepare Skills Temporary Git Boundary Design

## Summary

Add a temporary local git repository boundary to `SkillLinkManager.prepare_skills()` so staged operator workspaces always have a repo root at the workspace itself before backend skills are copied in.

If `workdir/.git` already exists, preserve it. If it does not exist, initialize a temporary repo with `git init` before skill staging and remove that `.git` during cleanup only when this run created it.

## Goals

- Ensure every `prepare_skills()` workspace has a local git boundary before skills are staged.
- Limit code-agent upward repository discovery to the operator workspace root even when the workspace lives under a larger parent repo.
- Preserve existing user-owned git repositories.
- Clean up only the temporary `.git` created by the current run.
- Keep the behavior shared across all existing `prepare_skills()` call sites.

## Non-Goals

- Do not detect whether the workspace is inside some ancestor git repo.
- Do not change backend-specific staged skill paths.
- Do not keep a temporary repo after the run finishes.
- Do not delete or rewrite any pre-existing `.git` directory or gitfile.

## User-Visible Behavior

Before staging backend skills, `prepare_skills()` checks only `workdir/.git`:

- If `workdir/.git` exists, skill staging proceeds unchanged.
- If `workdir/.git` does not exist, `prepare_skills()` runs `git init` in `workdir`.

After the agent run:

- If this run created `workdir/.git`, `cleanup()` removes it.
- If `workdir/.git` existed before staging, `cleanup()` leaves it untouched.

This behavior applies to every current caller of `prepare_skills()`, including generation, optimize, convert, log-check, and report flows.

## Safety Rules

- Treat only `workdir/.git` as the ownership boundary; do not inspect parent directories.
- Record whether the temporary repo was created by the current `prepare_skills()` call.
- If `prepare_skills()` fails after creating the temporary repo but before returning, remove the temporary `.git` before re-raising so failed staging does not leave a stray repo behind.
- If temporary `.git` cleanup fails, return a warning alongside existing skill cleanup warnings.

## Implementation Notes

- Extend `SkillLinkSet` to carry both staged skill copy paths and an optional temporary git path.
- Add a helper in `src/triton_agent/skills.py` that:
  - checks `workdir / ".git"`
  - runs `git init` when missing
  - raises a `RuntimeError` if initialization fails
- Wrap the existing `prepare_skills()` body so temporary repo creation rolls back on staging failure.
- Extend prepare/cleanup descriptions so verbose output mentions the temporary git boundary creation and removal.

## Testing

- Coverage proving `prepare_skills()` creates `workdir/.git` when missing and `cleanup()` removes it.
- Coverage proving an existing local `.git` is preserved and not recorded for cleanup.
- Coverage proving a workspace nested under a parent repo still gets its own local `.git`.
- Coverage proving a staging failure after temporary repo creation removes that `.git` before raising.
