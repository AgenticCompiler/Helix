# Skill Staging Refresh Design

## Summary

Change backend skill staging so a repeated `triton-agent` launch refreshes the
skill directories selected for the current run even when the backend-managed
staging root such as `.claude/skills` already exists.

## Goals

- Ensure updated repository skills are recopied on later launches instead of
  silently reusing stale staged directories.
- Limit the overwrite scope to the skills selected for the current run.
- Preserve existing cleanup guarantees and leave unrelated user-owned
  directories under the backend root untouched.

## Non-Goals

- Do not add skill version metadata, hashes, or manifest files.
- Do not change backend staging roots or copy-based staging strategy.
- Do not delete unrelated directories that were not selected for the current
  run.

## User-Visible Behavior

- On the first launch, skill staging behaves as it does today.
- On a later launch, if `.<backend>/skills/<skill-name>` already exists and the
  current run wants to stage that skill again, `triton-agent` removes that
  staged skill directory and copies the repository skill back in fresh.
- Directories under `.<backend>/skills/` that are not part of the current run,
  such as user-created helper folders, remain unchanged.

## Architecture

`src/triton_agent/skills/staging.py` should treat each selected staged skill
directory as a managed leaf. When `prepare_skills()` reaches a selected target
path that already exists as a normal directory, it should remove only that leaf
directory and then `copytree()` the source skill into place. Existing symlink
rejection behavior stays unchanged.

This keeps the overwrite boundary narrow: refresh only the explicitly selected
skill path, never the whole backend root.

## Testing

Add focused staging tests that cover:

- repeated staging refreshes an existing managed skill directory with new source
  contents
- repeated staging preserves unrelated sibling directories under the backend
  skills root
- symlink rejection still applies before any overwrite occurs
