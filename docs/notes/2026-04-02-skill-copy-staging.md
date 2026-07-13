# Skill Copy Staging

## Summary

- Replace workspace skill symlink creation with copy-based staging.
- Keep the user-visible workspace targets unchanged:
  - Codex uses `.codex/skills`
  - OpenCode uses `.opencode/skills/<name>`
- Treat the copied workspace skills as disposable run-owned content that is cleaned up only if created by the current run.

## User-Visible Behavior

- Before launching a code agent, the CLI copies this repository's skill content into the backend-specific workspace skill directory instead of creating symlinks.
- Copied skill files should appear to the code agent as ordinary workspace files, so agent-visible paths do not resolve back to the source repository through symlink expansion.
- Re-running the same command remains idempotent when the target workspace already contains matching copied skill directories from the current repository layout.
- If the target skill path already exists as a symlink or as unrelated user-owned content, the CLI should fail explicitly instead of silently reusing or replacing it.
- Cleanup should remove only the skill directories created by the current run.

## Implementation Notes

- Update `src/helix/skills.py` so the skill manager stages copies rather than symlinks.
- For Codex:
  - if `.codex/skills` does not exist, create it by copying the repository `skills/` tree
  - if `.codex/skills` already exists as a directory, copy only missing per-skill directories into it
- For OpenCode:
  - copy each skill directory into `.opencode/skills/<name>`
- Use `shutil.copytree(..., symlinks=False)` so staged skills are materialized as real files and directories.
- Record only paths created by the current run in `SkillLinkSet` so cleanup remains conservative.
- Keep cleanup safe:
  - remove run-owned directories recursively
  - never delete pre-existing content
  - never replace existing symlinks
- Update verbose messages to describe staged skill copies instead of skill links.

## Test Plan

- Update `tests/test_skills.py` to validate copied directories rather than symlinks.
- Add coverage for:
  - missing `.codex/skills`
  - existing `.codex/skills` directory with mixed pre-existing content
  - existing symlink target rejection
  - OpenCode per-skill copy staging
  - cleanup of only run-owned copied directories
- Run:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
