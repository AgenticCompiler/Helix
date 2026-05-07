# Benchmark Extra-Info Cleanup Design

## Summary

Local benchmark execution can leave an `extra-info/` directory behind in the benchmark working directory. That directory is scratch output and should not persist after `run-bench` completes.

## Goals

- Delete `extra-info/` from the local benchmark working directory after benchmark execution finishes.
- Apply the same cleanup rule to local `standalone` and local `msprof` benchmark modes.
- Keep cleanup narrowly scoped so benchmark runs do not touch unrelated files or directories.

## Non-Goals

- Do not change remote benchmark cleanup behavior beyond the existing remote workspace cleanup.
- Do not delete any path other than a directory named `extra-info` in the benchmark working directory.
- Do not remove profiler artifact directories that are intentionally preserved through `TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR`.

## Decision

- Define the local benchmark working directory as `bench_file.parent`.
- Run local standalone benchmarks under that working directory so both local benchmark modes share the same workdir contract.
- After each local benchmark run finishes, attempt to remove `bench_file.parent / "extra-info"`.
- If `extra-info/` does not exist, do nothing.
- If a filesystem entry named `extra-info` exists but is not a directory, leave it untouched.
- Run the cleanup in a `finally` path so it happens after both successful and failed local benchmark runs.

## Verification

- Add unit coverage that confirms local `msprof` benchmark runs delete `extra-info/` from the benchmark working directory.
- Add unit coverage that confirms local `standalone` benchmark runs also delete `extra-info/`.
- Add unit coverage that confirms a non-directory `extra-info` entry is preserved.
