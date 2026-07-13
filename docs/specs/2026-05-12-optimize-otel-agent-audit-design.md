# Optimize OTEL Agent Audit Design

## Summary

Optimize runs should create a per-run OTEL-style trace directory alongside the existing show-output log and agent session archive, then post-process that trace into `summary.json` and `agent-audit.md`.

## Context

The current optimize workflow already writes readable agent output to `helix-logs/optimize.show-output.log` when `--show-output` is enabled and records worker or supervisor session ids under `helix-logs/helix/<run-id>/agent-sessions.jsonl`. Those files are useful for manual diagnosis, but they do not provide stable structured facts for repeated file reads, staged skill script reads, repeated commands, or time attribution.

## Decision

Reuse the existing optimize run id from `helix-logs/helix/<run-id>/` as the OTEL run id and add:

```text
helix-logs/otel/<run-id>/trace.jsonl
helix-logs/otel/<run-id>/summary.json
helix-logs/otel/<run-id>/agent-audit.md
```

The first implementation records wrapper-level code-agent invocation events for every optimize worker, supervisor, and resume process. When backend hooks are enabled, Codex pre-tool hooks also append structured events for Bash tool calls, command classification, and file-read access visible through shell commands.

## Audit Rules

The post-processor reads `trace.jsonl`, `optimize.show-output.log`, `agent-sessions.jsonl`, and available round artifacts. It produces best-effort audit facts for:

- staged skill script reads
- repeated reads of the same file
- command categories and failed commands
- repeated full `msprof` benchmark commands
- remote SSH and benchmark time where duration evidence exists
- edit-test-fail loops where enough event evidence exists

Missing trace detail is reported explicitly instead of silently inventing evidence.

## Runtime Guidance

Optimize prompts should tell the worker not to read staged skill implementation scripts unless debugging, patching, or verifying that helper behavior, and to prefer `SKILL.md` and documented references for workflow guidance. Resume prompts should ask the agent to read the latest `agent-audit.md` when one exists before repeating previous actions.

## Boundaries

This change does not move optimize workflow logic out of skills. The CLI only stages hooks, records structured wrapper and hook events, and runs post-processing at the end of an optimize run. Detailed optimization decisions remain in skills and round artifacts.
