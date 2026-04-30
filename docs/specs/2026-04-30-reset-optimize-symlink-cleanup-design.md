# Reset Optimize Symlink Cleanup Design

## Goal

Make `--reset-optimize` safely clean known optimize artifacts even when one of those artifact paths is a symbolic link inside the workspace.

## Current Problem

`reset_optimize_workspace()` currently checks `path.is_dir()` before deciding between `shutil.rmtree()` and `Path.unlink()`. Directory symlinks report `is_dir() == True`, so reset attempts to call `rmtree()` on the symlink and fails with `OSError`.

## Desired Behavior

- Known optimize artifact symlinks should be removed as links with `unlink()`.
- Real directories should still be removed recursively with `rmtree()`.
- Real files should still be removed with `unlink()`.
- The change must stay scoped to `reset_optimize_workspace()` and must not widen cleanup beyond the existing recognized artifact set.

## Implementation Sketch

1. In `reset_optimize_workspace()`, check `path.is_symlink()` before `path.is_dir()`.
2. Keep the existing artifact list and round-directory discovery unchanged.
3. Add a regression test that creates symlinked optimize artifact directories and verifies reset removes only the links without raising.
