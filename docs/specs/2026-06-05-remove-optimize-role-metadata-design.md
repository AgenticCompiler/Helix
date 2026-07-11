# Remove Optimize Role Metadata Design

## Context

The optimize runtime still carries role metadata through several layers:

- `AgentRequest.optimize_role`
- optimize trace environment variables and trace events
- backend hook policy payloads
- optimize `agent-sessions.jsonl`
- prompt wording that frames invocations as roles

That metadata is not part of the user-facing optimize contract. The actual behavior is already determined by which prompt the CLI launches for a baseline pass, worker batch, or supervisor pass.

## Goals

- Remove request-level optimize role metadata.
- Remove role metadata from trace env, trace events, hook policy, and optimize session archives.
- Keep checked and supervised optimize behavior unchanged.
- Reword current prompts and guidance so they describe invocation purpose without using the term "role".

## Non-Goals

- Do not change checked versus supervised optimize control flow.
- Do not remove the supervisor pass itself.
- Do not redesign optimize archive layout beyond dropping role fields.

## Design

### Runtime Control Flow

The optimize runtime will continue to launch distinct baseline, round-batch, and supervisor invocations, but that distinction stays local to the call sites and prompt builders. It must not be stored on `AgentRequest` or emitted as generic metadata.

### Request Model

- Delete `AgentRequest.optimize_role`.
- Stop setting any replacement field for the same concept.

### Trace And Hooks

- Remove `HELIX_OTEL_ROLE`.
- Stop adding `role` to tool-trace env builders.
- Stop writing `role` into runner trace events, backend JSON trace adapters, and hook-emitted trace records.
- Remove `role` from backend hook policy structures.

### Optimize Session Archive

- Keep `agent-sessions.jsonl`.
- Reduce each record to `timestamp`, `session_id`, and `agent`.

### Prompt And Guidance Wording

- Replace wording such as "worker role" or "supervisor role" with "worker batch", "supervisor pass", or "this invocation".
- Keep shared guidance generic and prompt-driven, but describe the distinction as invocation-specific behavior rather than role-specific behavior.

## Verification

- Targeted unit tests for request modeling, optimize runtime, backend trace helpers, and optimize session archives.
- Full repo verification with:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
