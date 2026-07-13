## Summary

Remote Linux validation exposed that `process_runner` may send `SIGTERM` to an unrelated process group when a child was not launched in its own session. The buffered and PTY runners currently reap with `killpg(process.pid, SIGTERM)` even when `start_new_session=False`.

## Desired behavior

- Only use process-group signaling when the runner intentionally created and therefore owns that child process group.
- Fall back to direct `terminate()` / `kill()` when the child stayed in the caller's existing process group.
- Cover both buffered and PTY streaming paths with regression tests.

## Verification

- Run targeted local `pytest` coverage for `tests/test_process_runner.py`.
- Re-run remote `pytest` checks on `R154_cdj:/tmp` to confirm the previous `SIGTERM 143` failure is gone.
