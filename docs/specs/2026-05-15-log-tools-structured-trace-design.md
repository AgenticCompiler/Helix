# Log Tools Structured Trace

## User-Visible Semantics

`--log-tools` is an explicit opt-in for structured tool trace collection during agent-backed workflows. The first implementation targets `optimize` and `optimize-batch`.

`--log-tools` must not enable blocking workspace guard behavior by itself. `--enable-agent-hooks` controls guard behavior, while `--log-tools` controls trace collection. When both are enabled, the staged backend hook should both record trace events and enforce the guard policy.

## Initial Implementation Scope

- Add `--log-tools` to `optimize` and `optimize-batch`.
- Carry the option through `OptimizeRunOptions` and `AgentRequest`.
- Record L0 `agent_invocation` events for all backends when trace collection is enabled.
- Keep trace output under `helix-logs/otel/<run-id>/trace.jsonl`.
- Always write `summary.json` and `agent-audit.md` for optimize archive cleanup, with explicit trace capability and evidence-gap fields.
- Stage passive Codex/OpenCode trace hooks when `--log-tools` is enabled, without enabling guard denial unless `--enable-agent-hooks` is also enabled.

## Boundaries

The CLI owns trace setup, backend hook staging, and post-run audit output. Optimization workflow behavior remains in skills and round artifacts. Trace failures are fail-open and should not interrupt the agent workflow.
