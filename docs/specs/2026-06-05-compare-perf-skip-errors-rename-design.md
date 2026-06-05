## Summary

The run-eval MCP tool surface for `compare-perf` should expose `skip_error` instead of `skip_latency_errors`.

## User-Visible Behavior

Only the MCP tool schema should use the new name. The repository CLI and the underlying skill script keep their current `--skip-latency-errors` flag for now.

The MCP parameter should be described as skipping parse errors encountered while reading perf artifacts, not specifically latency errors.

## Implementation Notes

- Update only the MCP tool parameter name to `skip_error`.
- Keep the server-side translation to the existing `--skip-latency-errors` subcommand flag.
- Internal helper names and CLI flag names remain unchanged in this step.
