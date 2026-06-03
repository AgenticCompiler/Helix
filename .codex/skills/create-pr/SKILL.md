---
name: create-pr
description: Use when opening a new pull request for this repository requires the repo's branch, validation, and skill-script checking gates to run before PR creation.
---

# Create PR

Use this repo-local skill when opening a new PR for this repository. It only adds the repo-specific gates before handing off to `$managing-gitcode-prs`.

## When To Use

- Opening a new PR for the current repository
- Enforcing this repo's pre-PR checks and fresh-branch rule

Do not use this skill just to inspect, list, or update an existing PR. Use `$managing-gitcode-prs` for those tasks.

## Required Gates

- `GC_TOKEN` must be set before the final PR creation step.
- A new PR must come from a fresh topic branch. Do not open it from `main`, `master`, `dev`, or another reused branch.
- Finish with a clean working tree and a pushed branch.
- Run exactly these checks from the repository root:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
find skills -path '*/scripts/*.py' | sort | while IFS= read -r file; do
  bash scripts/run-skill-script-pyright.sh "$file"
done
```

Always run the skill-script loop for every `skills/*/scripts/*.py`, even when the current change did not touch any of those files. If any required command fails, stop, fix it, and rerun the full set before creating the PR.

## Final Handoff

Use `$managing-gitcode-prs` for the actual PR creation step. Keep GitCode-specific flags and API details there instead of duplicating them here.
