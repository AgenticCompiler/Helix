# GitCode PR Prune Source Branch Design

## Summary

Extend the `managing-gitcode-prs` skill and its bundled GitCode API helper so Codex can create pull requests that request source-branch deletion after merge.

## Motivation

The existing skill could create, list, and inspect GitCode pull requests, and it already supported post-create metadata updates for draft state. A user-facing gap remained: when the user asked for "delete source branch after merge", the skill had no documented flag or helper behavior for enabling that GitCode PR setting.

## Decision

- Add a `--prune-source-branch` flag to `.codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py`.
- Implement the flag as a follow-up `PATCH` request after PR creation, mirroring the existing draft-state update flow.
- Update the skill workflow guidance so Codex knows when to use the new flag.
- Update the command reference with an explicit example and behavior note.
- Extend the contract test so the checked-in skill helper must continue to expose this flag.

## Behavior

When `python3 <skill-path>/scripts/gitcode_pr_api.py create ... --prune-source-branch` is used:

1. The script creates the PR through the official GitCode pull request API.
2. The script reads the created PR number from the response.
3. The script issues `PATCH /pulls/{number}` with `{"prune_source_branch": true}`.
4. The final rendered output reflects the patched PR payload when the API returns updated metadata.

This keeps the CLI surface small while matching GitCode's API shape for post-create PR settings.

## Verification

- Run `uv run python -m unittest discover -s tests -v`.
- Run `uv run pyright`.
