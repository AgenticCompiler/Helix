# Benchmark Generation Contract

## Expected structure

- Import the operator explicitly.
- Build representative inputs.
- Warm up before measurement.
- Measure multiple iterations.
- Print or return a clear performance summary.

The full authoritative requirements live in:

- [bench-standalone-spec.md](bench-standalone-spec.md)
- [bench-msprof-spec.md](bench-msprof-spec.md)

## Standalone mode

- Focus on stable wall-clock timing.
- Report units and repeat counts.

## Msprof mode

- Keep the script simple and profiler-friendly.
- Avoid mixing extra logging into the hot path.

## Naming guidance

- Default output names can follow `bench_<operator>.py`.
