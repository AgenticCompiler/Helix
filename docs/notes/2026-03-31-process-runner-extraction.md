# Process Runner Extraction

## Summary

Move subprocess execution details out of `CodexRunner` so backend-specific code and local process management are easier to understand and reuse.

## User-Visible Behavior

- Command behavior should stay the same after the refactor.
- Buffered non-interactive runs should keep the same output collection and stall handling.
- `--show-output` should keep using the PTY-backed streaming path.

## Implementation Notes

- Keep codex-specific responsibilities in `CodexRunner`: command construction, verbose launch logging, and resume prompt handling.
- Move buffered pipe execution and PTY streaming execution into a dedicated `process_runner.py` helper.
- Test the extracted helper directly so process behavior and backend command behavior remain independently verifiable.
