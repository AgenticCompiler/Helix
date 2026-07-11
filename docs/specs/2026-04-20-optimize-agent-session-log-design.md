# Optimize Agent Session Log Design

## Summary

Record each optimize code agent launch in a run-level session log so later debugging can connect workspace artifacts to the backend session that produced them.

## Goals

- Persist the agent session id printed by backends such as Codex.
- Record one entry per code agent launch.
- Keep the record at optimize-run scope, not round scope, because one agent launch may create or revise multiple rounds.
- Keep the format small and easy to inspect.

## Non-Goals

- Do not add a large run metadata schema.
- Do not duplicate round-state, baseline-state, or optimize-status information.
- Do not place session ids under `opt-round-N/`.

## Storage

Write JSON Lines to:

`helix-logs/helix/<run-id>/agent-sessions.jsonl`

Each line contains only:

- `timestamp`
- `role`
- `session_id`
- `agent`

Example:

```json
{"timestamp":"2026-04-20T04:34:56Z","role":"worker","session_id":"019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5","agent":"codex"}
```

If the session id cannot be extracted, write `unknown`.

## Role Semantics

- Supervised worker launches use `worker`.
- Supervised supervisor launches use `supervisor`.
- Unsupervised optimize launches use `worker`, because that agent owns the optimize work.

## Implementation Notes

- Reuse the existing optimize log archive root and run id.
- Extend Codex session id extraction so it recognizes text output like `session id: <uuid>`.
- Append records after each `runner.run()` or `runner.resume()` call returns.
- Use best-effort logging: failure to write the session log should not mask the actual optimize result.

## Verification

- Unit-test Codex extraction from the startup text line.
- Unit-test supervised worker and supervisor entries.
- Unit-test unsupervised entries.
- Unit-test fallback to `unknown` when no session id is available.
