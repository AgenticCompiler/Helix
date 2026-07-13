# Profiler AIC Metrics Default Design

## Summary

Stop passing `aic_metrics=None` to Torch NPU's experimental profiler
configuration.

## User-Visible Behavior

- `run-bench` and `profile-bench` no longer emit the profiler message
  `Invalid parameter aic_metrics, reset it to default.`
- The profiler continues to use its runtime default AIC metrics setting.
- Existing profiler level, cache, and data-simplification settings remain
  unchanged.

## Rationale

The Ascend runtime reports that `None` is invalid and immediately resets it to
the same default used when the parameter is omitted. Omitting the argument
removes the warning without hiding diagnostics or changing the effective
configuration.

## Verification

- Add coverage that verifies `_ExperimentalConfig` receives no `aic_metrics`
  argument.
- Run focused benchmark tests, strict skill-script Pyright, and the repository
  validation commands.
