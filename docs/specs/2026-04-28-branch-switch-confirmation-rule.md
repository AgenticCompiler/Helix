# Branch Switch Confirmation Rule

## Summary

Add a durable repository rule that agents must not create or switch to a different git branch unless the user has explicitly confirmed that branch change.

## Motivation

Branch changes alter the user's working context and can have non-obvious consequences for local edits, pending work, and follow-up commands. This repository already treats `AGENTS.md` as the home for stable collaboration rules, so branch-switch safety belongs there.

## User-Visible Semantics

- Agents must stay on the current branch by default.
- Agents must ask for confirmation before creating or switching to another branch.
- This rule applies even when branch changes would otherwise be convenient for implementation workflow.

## Non-Goals

- Changing the repository's preferred branch naming conventions.
- Requiring confirmation for read-only git commands that do not change the current branch.

## Implementation

- Add a single bullet to `AGENTS.md` with explicit branch confirmation wording.
