# Optimize Per-Invocation Trace And Session Files

## Summary

Change optimize log artifacts so each agent launch writes its own trace file and its own session metadata file under the run archive directory.

This removes the optimize-only `otel/` subdirectory and stops appending multiple launches into one `agent-sessions.jsonl`.

Also simplify Claude human-readable `show-output` rendering by removing redundant tool ids from the displayed `[tool:start]` and `[tool:end]` lines.

## User-Visible Semantics

For one optimize run, the run archive root remains:

`triton-agent-logs/<run-id>/`

Within that directory, each agent launch owns its own files:

- baseline launch:
  - `show-output-baseline.log`
  - `trace-baseline.jsonl`
  - `trace-baseline.summary.json`
  - `agent-session-baseline.json`
- worker batch launch:
  - `show-output-batch-<start>-<end>.log`
  - `trace-batch-<start>-<end>.jsonl`
  - `trace-batch-<start>-<end>.summary.json`
  - `agent-session-batch-<start>-<end>.json`
- supervisor launch:
  - `show-output-supervisor.log`
  - `trace-supervisor.jsonl`
  - `trace-supervisor.summary.json`
  - `agent-session-supervisor.json`

`show-output` keeps its current behavior: one file per launch label, written during execution.

`--log-tools` keeps its current behavior: trace events are written during execution, and a summary file is generated after that launch completes.

For Claude backend `show-output` logs, tool lifecycle lines should stay human-oriented:

- keep tool name
- keep status, duration, return code, and excerpts
- do not print the internal `tool_use_id`

The underlying structured trace events must still retain `tool_use_id` for correlation and analysis.

## Goals

- Make optimize trace files line up with the existing per-launch `show-output` naming model.
- Make agent session metadata easy to map to one concrete launch without reading a JSONL append log.
- Keep one shared run archive directory so users can still inspect one optimize run as one bundle.
- Avoid summary-file collisions now that one optimize run may contain multiple trace files.
- Keep human-readable Claude `show-output` logs concise by removing backend-internal tool ids that do not help manual reading.

## Non-Goals

- Do not change non-optimize commands that still use one trace file per run.
- Do not redesign trace event schemas.
- Do not redesign optimize archive ownership or run-id generation.
- Do not upload trace files or agent session metadata through optimize upload.
- Do not remove `tool_use_id` from structured trace events.

## Path Rules

Optimize should derive trace and session file names from the same launch label already used by `show-output`.

When a launch has label `<label>`:

- trace path: `triton-agent-logs/<run-id>/trace-<label>.jsonl`
- trace summary path: `triton-agent-logs/<run-id>/trace-<label>.summary.json`
- agent session path: `triton-agent-logs/<run-id>/agent-session-<label>.json`

There is no optimize `otel/` directory after this change.

## Runtime Flow

At optimize request preparation time, the run archive directory is still chosen once per optimize invocation.

At each individual agent launch:

1. The controller already knows the launch label such as `baseline`, `batch-1-5`, or `supervisor`.
2. It uses that label to set the launch-local trace path in request env.
3. The runner writes trace events only to that file for that launch.
4. After the launch returns, optimize writes one compact JSON object for that launch's session metadata file.
5. Trace summary generation writes the matching `trace-<label>.summary.json`.

No optimize launch appends trace or session metadata into another launch's file.

## Claude Show-Output Rendering

This change is limited to the human-readable `show-output` text emitted by the Claude backend renderer.

Current rendered lines include the internal tool id, for example:

- `[tool:start] Read call_abc123`
- `[tool:end] Read call_abc123 ok in 120ms rc=0`

After this change, the rendered lines should become:

- `[tool:start] Read`
- `[tool:end] Read ok in 120ms rc=0`

All structured trace events should continue to record `tool_use_id` exactly as they do now so tool start/end correlation and downstream analysis remain intact.

## Session Metadata Format

Each `agent-session-<label>.json` file stores one JSON object with the current compact fields:

- `timestamp`
- `session_id`
- `agent`

If no session id is available, `session_id` remains `unknown`.

## Summary File Naming

Optimize needs one summary file per trace file.

For optimize trace files named `trace-<label>.jsonl`, the summary file should be `trace-<label>.summary.json`.

Existing non-optimize paths should keep their current `summary.json` behavior:

- `tool-traces.jsonl` -> `summary.json`
- `trace.jsonl` -> `summary.json`

This keeps existing command behavior stable while preventing optimize summary collisions.

## Upload Impact

Optimize upload already excludes `agent-sessions.jsonl`.

After this change, upload should exclude any `agent-session-*.json` files as the optimize-session metadata equivalent.

Trace files remain excluded because upload only includes `show-output*.log` from `triton-agent-logs/`.

## Compatibility Notes

- Existing historical run directories may still contain `otel/trace.jsonl` or `agent-sessions.jsonl`.
- New optimize runs should not create those files.
- `trace-analyze` should continue to work when pointed at any individual trace file path.

## Verification

- Add or update unit tests for optimize archive path construction.
- Add or update optimize runtime tests to verify one session file per launch instead of appended JSONL.
- Add or update trace summary path tests so optimize trace files produce `trace-<label>.summary.json`.
- Run targeted unittest coverage for optimize runtime, optimize guidance, trace analyze, and backend trace path handling.
