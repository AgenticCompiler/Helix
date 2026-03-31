# Show Output Streaming

## Summary

Make `--show-output` display agent output as it is produced instead of only after process completion.

## User-Visible Behavior

- In non-interactive mode, `--show-output` should stream agent output to the current terminal in real time.
- The CLI should not print the same output again at the end of the run.
- The behavior should stay readable and compatible with verbose diagnostics.

## Implementation Notes

- Use a PTY-backed execution path for streamed output so the child process behaves more like it is attached to a terminal.
- Keep the normal buffered execution path for non-streaming runs.
- Preserve aggregated output in the returned result object so supervisor and post-run handling continue to work.
