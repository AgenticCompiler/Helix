# Bench Runner And Standalone Runtime Refactor Design

## Summary

`bench_runner.py` and `standalone_bench_runtime.py` currently duplicate benchmark metadata parsing, kernel resolution, perf artifact rendering, and some profile-directory helpers. `standalone_bench_runtime.py` also carries a `main()` entrypoint that is no longer needed once `profile-bench` owns standalone execution.

## Goals

- Extract shared benchmark-contract helpers out of both runtime files.
- Extract shared perf-artifact parsing, rendering, and comparison logic into an independent module.
- Remove the standalone script entrypoint and treat `standalone_bench_runtime.py` as a library module.
- Make `profile-bench` the only place that decides how standalone profiling is launched locally and remotely.

## Non-Goals

- Do not change perf artifact semantics or file formats.
- Do not change benchmark case selection behavior.
- Do not add new user-facing CLI flags.

## Decision

- Move benchmark metadata parsing and Triton kernel resolution into a shared helper module.
- Move perf artifact formatting/comparison into a shared helper module.
- Keep `bench_runner.py` and `standalone_bench_runtime.py` focused on orchestration and mode-specific execution.
- Replace any direct `python3 standalone_bench_runtime.py ...` launch path with module-level calls from `profile_runner.py`.
- Remove the standalone `main()` wrapper once no caller depends on it.

## Verification

- Add or update tests so local and remote `profile-bench` still work for `standalone` and `msprof`.
- Add or update perf comparison tests so the extracted perf helper remains the source of truth.
- Run focused unit tests for bench execution, profile execution, and perf comparison after the refactor.
