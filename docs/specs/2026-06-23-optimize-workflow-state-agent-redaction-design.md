# Optimize Workflow State Agent Redaction Design

## Goal

Prevent the temporary optimize workflow state file `.helix/state.json` from being exposed to code agents through runtime-owned surfaces.

## Scope

This change covers only agent-facing runtime surfaces:

- optimize prompts
- temporary workspace guidance files such as `AGENTS.md` and `CLAUDE.md`
- tool denial messages and other hook-managed tool results
- optimize skill script JSON payloads that are returned to the agent
- runner-managed hidden runtime and hook files that would otherwise reveal the protected path if the agent reads them directly

This change does not rewrite historical human-authored design or plan documents in the repository.

## Design

Treat `.helix/state.json` as runner-private state rather than agent-visible workspace context.

1. Prompt redaction:
   Remove the concrete workflow-state path from the rendered phase summary. The prompt should still describe the current phase, active round, and baseline status because the agent needs that guidance, but it should not learn the backing filename.

2. Tool-result redaction:
   Replace agent-facing workflow-state recovery text that currently names `.helix/state.json` with generic wording such as "temporary optimize workflow state" or "runner-managed workflow state."

3. Read protection:
   Expand the read-deny policy so agents cannot inspect runner-managed hidden runtime files that would reveal the protected state path indirectly. This includes the live `.helix/` runtime tree and staged hook internals in backend-specific hidden directories.

4. Workflow helper sanitization:
   Ensure workflow helper exceptions and optimize submit/start script payloads do not echo the protected path back in their `issues` or `guideline` fields.

5. Visible agent-authored handoff files:
   Files that the agent is expected to read or write should not live under the hidden runner-managed `.helix/` tree. The supervisor handoff report should therefore use a top-level `supervisor-report.md` path while `.helix/` remains reserved for runner-private state such as workflow tracking and history snapshots.

## Non-Goals

- moving workflow state out of the workspace
- changing optimize workflow semantics
- scrubbing historical docs that already mention the path for human readers
