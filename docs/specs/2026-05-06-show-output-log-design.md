# Show Output Log Design

## Summary

When `--show-output` is enabled for a non-interactive code-agent run, keep streaming readable agent output to the current terminal and also append the same output to a persistent workspace log file for debugging.

## Goals

- Preserve the current `--show-output` terminal behavior.
- Persist streamed code-agent output under the workspace that owns the run.
- Keep retry attempts distinguishable in the saved log.
- Reuse one shared implementation path for CLI-backed code-agent backends.

## Non-Goals

- Do not add a new CLI flag or log-path option in this change.
- Do not change interactive mode behavior.
- Do not redesign optimize session archives or replace existing `optimize-logs/` artifacts.

## Storage

For each workspace-scoped agent request, append to:

`<workdir>/triton-agent-logs/<command-kind>.show-output.log`

Examples:

- `.../triton-agent-logs/gen-test.show-output.log`
- `.../triton-agent-logs/convert.show-output.log`
- `.../triton-agent-logs/optimize.show-output.log`

Batch commands still write one log per workspace because each workspace already has its own request workdir.

## Log Format

Append UTF-8 text with lightweight attempt markers:

- a start marker before each actual agent launch attempt
- the streamed readable agent output
- an end marker with return code, stalled flag, and session id fallback

This keeps retries and repeated runs inspectable without inventing a larger structured schema.

## Implementation Notes

- Add a small shared helper that opens the workspace log in append mode and writes attempt markers.
- Extend the shared process runner streaming path so it can duplicate filtered output to both the terminal stream and a raw log stream.
- Have the shared backend runner own log-session lifecycle so all CLI-backed backends inherit the behavior automatically.
- Update the OpenHands backend separately because it bypasses the shared process runner.

## Verification

- Unit-test that streaming output is duplicated to a separate log stream without polluting the terminal prefixing behavior.
- Unit-test that shared backend retries append multiple attempt markers into one log file.
- Unit-test that OpenHands `--show-output` also writes its event text into the workspace log.
