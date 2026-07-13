# Streaming PTY EIO Race Design

## Goal

Prevent OpenCode and other PTY-backed non-interactive backends from crashing with `OSError: [Errno 5] Input/output error` during normal child shutdown.

## Problem

The shared PTY streaming runner currently treats `OSError(errno.EIO)` as clean EOF only when `process.poll()` already reports that the child exited. On Linux PTYs, the master side can raise `EIO` slightly before `poll()` observes the exit, so a normal shutdown can still escape as a traceback.

## User-Visible Behavior

- `--show-output` runs that use PTY streaming should exit cleanly when the backend process finishes, even if PTY EOF arrives as `EIO` before `poll()` updates.
- Real PTY read failures should still surface instead of being silently swallowed.
- Backend-specific command construction should remain unchanged.

## Design

Keep the fix inside `src/helix/process_runner.py`.

- When `os.read()` on the PTY master raises `OSError(errno.EIO)`, first keep the existing fast path for already-exited children.
- If `poll()` still returns `None`, give the child a short exit-confirmation grace window with `wait(timeout=...)`.
- Treat the `EIO` as EOF only if that confirmation observes process exit; otherwise re-raise the original `OSError`.

This keeps the normal Linux PTY shutdown race on the clean path without broadening `EIO` handling into a blanket ignore.

## Testing

- Add a regression test where `os.read()` raises `EIO` before `poll()` reports exit, but a short `wait(timeout=...)` confirms the child has finished.
- Preserve coverage that non-racy `EIO` after child exit remains clean.
- Preserve the behavior that non-EOF PTY read failures still raise.

## Scope

- Do not change backend command assembly.
- Do not change buffered or interactive process runners.
- Do not alter stall-timeout behavior outside the PTY EOF path.
