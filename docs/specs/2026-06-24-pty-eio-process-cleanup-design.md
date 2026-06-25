# PTY EIO Process Cleanup

## Summary

Prevent leaked agent subprocesses when the Unix PTY streaming runner sees an unexpected `EIO` before the child process exits.

## Problem

`optimize-batch` bounds logical workspace concurrency with a thread pool, but the PTY streaming runner currently raises immediately when `os.read()` returns an `EIO` and the child process still appears alive after the short EOF grace window.

That exception ends the batch future, so the executor may schedule the next workspace even though the previous agent process is still running in the background. For `optimize-batch --agent claude`, this can make real operator activity exceed the requested `--concurrency` limit.

## User-Visible Semantics

- Batch commands should not leave background agent processes running after the wrapper has already marked that workspace invocation as failed.
- An unexpected PTY transport break should keep surfacing as a failure, but the wrapper must clean up the associated process tree first.

## Design

- Keep the existing PTY EOF detection logic.
- When PTY reads fail with `EIO` and the child process does not exit within the existing grace window:
  - terminate the child process tree before propagating the error
  - reuse the existing Unix process-group cleanup helper so optimize workers started in their own session are fully reaped
- Do not change normal successful PTY EOF handling.

## Verification

- Add process-runner coverage proving abnormal PTY `EIO` cleanup terminates the Unix process group before the error escapes.
- Re-run focused process-runner tests plus repository lint, type checking, and the full test suite.
