# Friendly CLI Errors

## Summary

Expected user-facing command problems should be reported as concise CLI errors, not Python tracebacks.

## User-Visible Behavior

- When a generation target already exists and overwrite is not allowed, the CLI should print a short error message and exit with a non-zero status.
- The message should explain what happened and how to proceed, for example by using `--force-overwrite`.
- The CLI should avoid exposing Python traceback details for this class of expected local validation error.

## Implementation Notes

- Keep traceback suppression limited to expected validation failures that are part of normal CLI usage.
- Preserve normal exception behavior for unexpected internal failures so they remain debuggable during development.
