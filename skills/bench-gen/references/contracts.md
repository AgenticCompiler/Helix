# Benchmark Generation Contract

## CLI interface

All generated benchmark files must accept `--operator-file` and `--api-name` CLI arguments and use `importlib` to dynamically load the operator. Msprof mode additionally requires `--bench <N>` and `--num-bench`. See the spec files for the full entry point pattern.

## Expected structure

- Load the operator via `--operator-file` and `--api-name` (dynamic import).
- Build representative inputs.
- Warm up before measurement.
- Measure multiple iterations.
- Print or return a clear performance summary.

## Authoritative specs

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
