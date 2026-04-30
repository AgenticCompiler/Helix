# GitCode PR Skill Design

## Summary

Add a repo-local Codex skill for GitCode pull request workflows that only covers `gc pr create`, `gc pr list`, and `gc pr view`.

## Motivation

This repository already treats skills as the primary place for workflow guidance. The requested behavior is not a generic GitCode CLI skill; it is a focused workflow helper for opening and inspecting pull requests from the current project. Keeping the scope narrow avoids teaching unrelated `gc` commands and keeps the trigger surface aligned with PR tasks only.

## User-Visible Semantics

- The skill lives under `.codex/skills/` so it can be used from this project workspace.
- The skill triggers when Codex needs to create, list, or view GitCode pull requests for the current repository.
- The skill requires `GC_TOKEN` to be present in the environment before running `gc pr` commands.
- For this repository, the skill defaults to `-R midwinter1993/triton-agent` unless the user explicitly targets another repo.
- The skill teaches safe defaults for current-branch and current-repository workflows, while documenting explicit fallback flags such as `--head`, `--base`, and `-R`.
- The skill prefers structured `--json` output for follow-up inspection tasks and avoids browser-opening flags unless the user explicitly asks for them.

## Scope Boundaries

- In scope:
  - `gc pr create`
  - `gc pr list`
  - `gc pr view`
  - Environment precheck for `GC_TOKEN`
  - Guidance for current branch detection and explicit `--head` fallback
- Out of scope:
  - Other GitCode CLI areas such as issues, auth, repo management, or CI
  - General Git workflows beyond the minimum branch context needed for PR commands
  - Project CLI changes in `src/`

## Skill Shape

- Create `.codex/skills/managing-gitcode-prs/`
- Add `SKILL.md` with concise workflow instructions in English
- Add `agents/openai.yaml` for skill list metadata
- Add `references/pr-command-reference.md` with the command patterns and option notes needed to keep `SKILL.md` compact

## Validation

- Run the skill initializer and then replace the template contents with the final skill text.
- Run the skill validator on `.codex/skills/managing-gitcode-prs/`.
- Review the resulting files to ensure the skill remains PR-only and does not drift into a generic GitCode CLI guide.
