# Optimize Progress Probe Transient Path Handling

## User-visible behavior

- `optimize` must not crash just because operator validation or compilation creates and removes transient workspace directories such as `kernel_meta/`.
- Progress detection should continue to notice real optimize progress from business artifacts only:
  - `opt-note.md`
  - `learned_lessons.md`
  - files under `baseline/`
  - files under `opt-round-* /`
- Transient, unrelated workspace paths must not affect stall detection or terminate the agent process.

## Implementation direction

- Stop scanning the entire workspace tree with `rglob("*")` for optimize progress probing.
- Enumerate only the allowlisted progress roots already implied by `is_optimize_progress_path`.
- Walk `baseline/` and `opt-round-* /` defensively so disappearing files or directories are skipped instead of raising `FileNotFoundError`.
- Preserve the existing fingerprint shape `(relative_path, size, mtime)` so recovery and stall logic remain unchanged.

## Verification

- Add a regression test that simulates a transient non-progress path disappearing during scan and confirms `scan_optimize_progress()` still returns the expected progress snapshot.
