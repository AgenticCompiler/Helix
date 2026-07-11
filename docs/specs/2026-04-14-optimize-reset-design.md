# Optimize Reset Design

## Summary

- Add `--reset-optimize` to `optimize` and `optimize-batch`.
- Require `--reset-optimize` to be used only with `--resume fresh`.
- Remove optimize-session artifacts before fresh-mode workspace validation.
- Preserve reusable validation harnesses so fresh optimize runs can keep existing tests and benchmarks.

## Behavior

- `--resume fresh` without `--reset-optimize` keeps the current behavior.
- `--resume fresh --reset-optimize` deletes only known optimize-session artifacts:
  - `opt-note.md`
  - `learned_lessons.md`
  - `baseline/`
  - `opt-round-*`
  - `.helix/`
  - `helix-logs/helix/`
  - default `opt_<operator>.py`
- `--reset-optimize` does not delete reusable validation harnesses:
  - `test_<operator>.py`
  - `differential_test_<operator>.py`
  - `bench_<operator>.py`

## Resume Semantics

- After reset cleanup, fresh-mode validation should treat preserved harnesses as reusable inputs, not as blocking optimize-session artifacts.
- Fresh mode should still fail when optimize-session artifacts remain after reset cleanup.
- Fresh mode continues to use explicit `--test-mode` and `--bench-mode` when provided; otherwise it keeps the existing fresh defaults.

## Error Handling

- `--reset-optimize` without `--resume fresh` fails with a short actionable error.
- Cleanup is limited to recognized optimize-session artifacts and must not delete unrelated user files.
