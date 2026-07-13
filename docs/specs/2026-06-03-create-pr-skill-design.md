# Create-PR Skill Design

## Summary

Add a repo-local skill named `create-pr` that gates pull request creation for this repository behind a required validation workflow, mandatory fresh branch creation, and reuse of the existing `managing-gitcode-prs` skill for the final GitCode PR API call.

## Motivation

This repository already treats skills as the primary place for workflow guidance. Opening a PR here is not just an API action: it also requires repository-specific checks from `AGENTS.md`, an extra strict type-checking pass for skill-side Python scripts, and a clear rule that a PR must come from a newly created topic branch instead of a long-lived branch. Encoding that flow in a repo-local skill keeps the policy close to the project and avoids pushing workflow logic into `src/`.

## User-Visible Semantics

- The skill lives at `.codex/skills/create-pr/`.
- The skill triggers when an agent needs to create a pull request for this repository rather than merely inspect existing PRs.
- The skill requires a newly created topic branch before PR creation, but does not require any specific branch-name prefix.
- The skill requires the repository validation commands for this workflow:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
- The skill also requires strict file-scoped Pyright checks for every Python file under `skills/*/scripts/` by using the existing helper script:
  - `bash scripts/run-skill-script-pyright.sh <skill-script.py>`
- The skill stops immediately when any required validation step fails.
- The skill requires the new branch to be pushed before PR creation.
- The skill delegates the final PR creation step to the existing repo-local `managing-gitcode-prs` skill instead of duplicating GitCode API guidance.

## Scope Boundaries

- In scope:
  - repo-local pull request submission workflow
  - required branch creation rule for new PRs
  - repository validation command checklist
  - strict checking of all skill-side Python scripts
  - handoff to `managing-gitcode-prs` for actual PR creation
- Out of scope:
  - changes under `src/helix/`
  - CLI subcommands for PR creation
  - generic Git tutorials
  - replacing or expanding the existing `managing-gitcode-prs` API skill

## Skill Shape

- Create `.codex/skills/create-pr/`.
- Add `SKILL.md` with the durable workflow instructions in English.
- Add `agents/openai.yaml` for repo-local skill discovery metadata.
- Keep the skill self-contained; avoid a separate command reference file unless the workflow later grows beyond a short inline checklist.
- Do not add scripts unless the workflow later proves too repetitive to keep as plain instructions.

## Workflow Contract

1. Inspect the current git state and determine whether the task is to open a new PR for the current repository.
2. Require creation of a fresh topic branch before continuing. The branch name may follow any user- or team-chosen convention.
3. Run the required repository validation commands for this workflow.
4. Enumerate every `skills/*/scripts/*.py` file in the repository and run `bash scripts/run-skill-script-pyright.sh <file>` for each one.
5. If any command fails, stop and surface the failure instead of continuing to branch push or PR creation.
6. Ensure the branch has commits and has been pushed to the remote.
7. Use the repo-local `managing-gitcode-prs` skill to create the PR.

## Command Presentation

The skill should only show the commands that are repo-specific or easy to forget:

- the required validation commands, including the pytest invocation
- the loop that checks every `skills/*/scripts/*.py` file with `scripts/run-skill-script-pyright.sh`

Do not pad the skill with generic git command templates or repeated PR API examples that already live in `managing-gitcode-prs`.

## Implementation Notes

- This is a repo-local skill only, so `src/helix/skill_staging.py` should remain unchanged.
- The skill should reference `AGENTS.md` for the static-analysis command expectations and for the repository rule that agent-run pytest commands must use `-q --tb=short --no-header -p no:warnings`.
- The final PR step should explicitly point readers to the existing `.codex/skills/managing-gitcode-prs/` workflow for GitCode-specific flags such as `--draft`, `--fill`, or explicit repo and branch overrides.

## Verification

- Initialize the skill skeleton under `.codex/skills/create-pr/`.
- Validate the resulting skill with the skill validator.
- Review the final skill content to ensure it enforces:
  - new-branch-before-PR semantics without a forced prefix
  - the required `ruff`, `pyright`, and pytest validation commands
  - strict checks for every `skills/*/scripts/*.py`
  - delegation to `managing-gitcode-prs` for the final PR creation step
