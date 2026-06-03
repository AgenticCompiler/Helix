# Incremental Show-Output Streaming Log

## Summary

Change non-interactive `--show-output` runs so rendered agent output is appended to the workspace show-output log incrementally while the agent is still running, instead of buffering the entire stream in memory and writing it only after the process exits.

## Goals

- Make the `triton-agent-logs/*.show-output.log` file durable during execution so unexpected agent or wrapper termination still leaves partial output behind.
- Keep the log file content identical to the rendered agent output stream.
- Remove wrapper-added show-output markers, counters, and attempt summaries from the log file.
- Stop treating `AgentResult.stdout` as the canonical storage for streamed `--show-output` output.
- Keep retry behavior working for transient agent failures without rescanning accumulated streamed output text.

## Non-Goals

- Do not change buffered non-`--show-output` execution behavior.
- Do not change interactive execution behavior.
- Do not change the readable rendering format produced by backend-specific output filters such as Codex JSON or Claude stream-json.
- Do not add new wrapper text to the show-output log to separate retries or attempts.

## Design

### 1. Show-output log content becomes a direct rendered stream

For non-interactive `--show-output` runs, the show-output log remains the same file path returned by `show_output_log_path()`, but its content becomes a direct append-only copy of the rendered output stream.

The log file must contain only the agent-visible rendered text:

- filtered readable text for backends that use an `OutputFilter`
- raw stdout text for backends without an output filter
- OpenHands event text rendered by the SDK adapter

The wrapper must not prepend or append:

- attempt start or end markers
- event, tool, or error counters
- session-id footer lines
- any other wrapper-specific annotations

If a retry occurs, later output is simply appended after earlier output with no synthetic boundary line.

### 2. Streaming execution writes each rendered chunk immediately

The streaming runner path should accept an optional incremental sink or callback for rendered chunks.

For each decoded process chunk:

1. Decode bytes into text.
2. Run the backend `output_filter` if one exists.
3. Print the rendered text to the live terminal stream exactly as today.
4. Append the same rendered text to the show-output log immediately.
5. Flush the log stream immediately.

This change applies only to streaming `--show-output` execution. Buffered execution may continue collecting stdout in memory and returning it in `AgentResult.stdout`.

### 3. Streaming `AgentResult.stdout` is no longer the full log payload

For the show-output streaming path, `AgentResult.stdout` should no longer contain the full rendered output. The canonical persisted output is now the show-output log file itself.

Expected result semantics:

- streaming `--show-output`: `result.stdout == ""`
- buffered non-`--show-output`: preserve current `result.stdout`
- OpenHands with `show_output=True`: mirror streaming behavior and return empty stdout
- OpenHands with `show_output=False`: preserve current aggregated stdout behavior

This keeps the memory win local to the streaming path while minimizing unrelated behavior changes.

### 4. Retry detection moves from whole-output scanning to explicit result metadata

Current retry detection scans `result.stdout + result.stderr` after the run completes. That no longer works once streaming `stdout` is intentionally empty.

Add an explicit boolean field on `AgentResult` for transient retry detection, for example `retryable_failure`.

For streaming runs:

- maintain a small rolling raw-text window while chunks arrive
- match transient failure patterns against that rolling window
- set the result field when a match is observed

Detection should happen on decoded raw process text before readable filtering so backend-specific renderers do not accidentally hide retry keywords.

`AgentRunner` retry logic should then prefer the explicit result field and fall back to the existing stdout/stderr scan only for non-streaming paths.

### 5. Continuous optimize stall recovery stops summarizing captured stdout

`OptimizeRunLoop` currently derives a resume summary from captured stdout/stderr for stalled recovery attempts. That couples recovery behavior to the in-memory streamed output buffer.

For stalled recovery retries:

- stop building summaries from `result.stdout`
- continue using the resume path so backends keep their existing session-continuation behavior
- pass a fixed workspace-based continuation summary instead of replaying captured output text

Example intent:

`The previous invocation ended unexpectedly before completion. Continue from the existing workspace state and recorded optimize artifacts.`

The existing minimum-round continuation summary built from workspace round counts should remain unchanged because it is not derived from streamed output.

### 6. User-facing failure details should not depend on streamed stdout

Callers that currently use `result.stdout.strip()` as a fallback failure detail need to stop depending on it for `show_output=True` paths.

For streaming show-output failures, prefer:

- `result.stderr` when present
- otherwise a generic failure message that points users to the workspace show-output log path

This especially affects report and log-check entry points that currently build failure summaries from `stdout`.

### 7. Agent invocation OTEL events should stop duplicating stdout/stderr excerpts

The `agent_invocation` trace event currently records stdout/stderr digests and excerpts even though the full human-readable stream already lives in the show-output log.

Remove the `agent_invocation`-level duplicated output payload fields:

- `stdout_digest`
- `stderr_digest`
- `stdout_excerpt`
- `stderr_excerpt`

Keep the useful structured metadata such as timing, status, return code, command kind, backend, role, and session id.

## Verification

- Add or update process-runner tests to prove a streaming sink receives rendered text incrementally and that streamed results can return empty stdout.
- Add or update backend-base tests to prove show-output logs contain only streamed agent text with no wrapper markers.
- Add or update OpenHands tests to prove SDK event text is written incrementally and that `show_output=True` no longer returns aggregated stdout.
- Update optimize recovery tests to prove stalled recovery no longer reuses captured stdout as the resume summary.
- Update trace tests only as needed for the removal of `agent_invocation` stdout/stderr excerpt fields.
