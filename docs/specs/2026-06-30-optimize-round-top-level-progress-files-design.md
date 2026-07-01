## Summary

During an active optimize round, the built-in edit guard should continue blocking arbitrary top-level workspace edits, but it should allow the small set of top-level progress files that the optimize workflow treats as session-wide artifacts.

## Problem

The current active-round guard only allows built-in edits inside the active `opt-round-N/` directory. This conflicts with the optimize workflow, which treats several top-level files as durable session artifacts that may need incremental updates while a round is in progress.

That mismatch causes the agent to be told to maintain top-level optimize records, then get blocked when it tries to do so.

## Policy

When workflow phase is `round_active`, built-in edits remain allowed for:

- anything inside the active `opt-round-N/`
- `opt-note.md`
- `learned_lessons.md`
- `supervisor-report.md`

All other top-level edits should remain blocked, including the source operator file and unrelated workspace files.

## Scope

This change is limited to the built-in edit guard policy used by Codex and OpenCode wrappers.

It does not broaden read policy, baseline-phase policy, or shell-write policy.
