# Streaming PTY Exit Cleanup

## Summary

Treat PTY `EIO` on child shutdown as a normal end-of-stream for non-interactive streamed agent runs.

## User-Visible Behavior

- `--show-output` should stream output live and then exit cleanly when the agent process finishes.
- The CLI should not print a Python traceback during normal PTY teardown on platforms where PTY EOF is reported as `EIO`.
- Real PTY read failures should still surface as errors.

## Implementation Notes

- Keep the PTY-backed streaming path for readable live output.
- In the streaming read loop, treat `OSError(errno.EIO)` as EOF once the child exit is confirmed, either because `poll()` already reports termination or because a short `wait(timeout=...)` grace window observes the exit.
- Add regression coverage for both the normal shutdown race and the path where a real PTY read failure must still surface.
