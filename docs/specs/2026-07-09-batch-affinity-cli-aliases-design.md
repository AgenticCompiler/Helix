# Batch Affinity CLI Aliases Design

## Summary

Add singular long-option aliases for the existing batch-affinity CLI flags
without changing their internal field names, environment fallback behavior, or
runtime semantics.

## User-Visible Semantics

- `--npu-devices` also accepts `--npu-device`.
- `--workers-per-npu` also accepts `--worker-per-npu`.
- Both spellings populate the existing argparse destinations:
  `npu_devices` and `workers_per_npu`.
- No other CLI behavior changes.

## Implementation Shape

- Update the shared argparse registration in [src/triton_agent/cli.py](/Users/cdj/Projects/triton-agent/src/triton_agent/cli.py)
  so both the direct `--npu-devices` path and the batch-affinity path expose
  the singular alias.
- Keep the change parser-only; do not touch downstream command handling.
