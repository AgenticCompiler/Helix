# Unified Process Runner Modes

## Summary

Route interactive, buffered, and streaming execution through one process-runner entrypoint so backend adapters stay focused on backend-specific behavior.

## User-Visible Behavior

- Interactive behavior should remain unchanged.
- Buffered and streaming non-interactive behavior should remain unchanged.
- The refactor should not alter CLI semantics; it only simplifies internal ownership.

## Implementation Notes

- Add a `run_process(...)` entrypoint that dispatches to interactive, buffered, or streaming execution.
- Keep mode selection in backend adapters such as `CodexRunner`.
- Keep subprocess lifecycle details in `process_runner.py` so future backends can reuse the same local execution helpers.
