# Torch NPU Owner Warning Suppression Design

## Summary

Suppress the known `torch_npu.utils.collect_env` `UserWarning` that reports a
CANN directory owner mismatch while run-eval commands initialize Torch NPU.

## User-Visible Behavior

- `run-test`, `run-bench`, and `profile-bench` no longer emit this specific
  owner-mismatch warning in local or remote execution.
- The warning is matched by its `torch_npu.utils.collect_env` source module,
  `UserWarning` category, and owner-mismatch message.
- Other warnings, stderr output, command results, and exit codes are
  unchanged.

## Implementation

- Add a small self-contained runtime helper that installs the precise Python
  warning filter before Torch NPU initialization.
- Use it in the test and benchmark runtime bootstraps.
- Embed the equivalent standard-library filter in self-contained remote test
  commands, and stage the helper with remote benchmark runtime dependencies.

## Verification

- Add unit coverage that proves the target warning is suppressed and unrelated
  warnings remain visible.
- Check that the helper is installed by both local runtime bootstraps and is
  included in the remote benchmark support files.
- Run focused tests, the required skill-script Pyright checks, and repository
  lint/type/test validation.
