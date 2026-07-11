# Optimize Interact Validation Regression Design

## Goal

Restore `helix optimize --interact` for supported backends.

## Problem

The optimize runtime still supports interactive worker invocations, but the command-layer
validation now rejects every `optimize --interact` request with the batched-round-mode error.
This blocks previously supported interactive optimize usage even for single-workspace runs.

## User-Visible Behavior

- `helix optimize --interact ...` should be accepted for supported backends.
- `helix optimize --agent openhands --interact ...` should remain rejected.
- `optimize-batch` remains non-interactive.

## Implementation Notes

- Narrow the optimize command validation so the batched-mode `--interact` rejection applies
  only to `optimize-batch`, not single-workspace `optimize`.
- Keep the existing runtime behavior where interactive mode stays attached to worker
  invocations while supervisor/audit invocations stay non-interactive.
