# Benchmark Extra-Info Cleanup Design

## Summary

Local benchmark execution can leave an `extra-info/` directory behind in the benchmark working directory. That directory is scratch output and should be deleted after each benchmark case finishes, not only after the full run completes.

## Goals

- Delete `extra-info/` from the local benchmark working directory after each local benchmark case finishes.
- Apply the same cleanup rule to local `standalone` and local `msprof` benchmark modes.
- Keep cleanup narrowly scoped so benchmark runs do not touch unrelated files or directories.

## Non-Goals

- Do not change remote benchmark cleanup behavior beyond the existing remote workspace cleanup.
- Do not delete any path other than a directory named `extra-info` in the benchmark working directory.
- Do not remove profiler artifact directories that are intentionally preserved through `TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR`.

## Decision

- Define the local benchmark working directory as `bench_file.parent`.
- Run local standalone benchmarks under that working directory so both local benchmark modes share the same workdir contract.
- In local `msprof` mode, attempt to remove `bench_file.parent / "extra-info"` in the per-case loop after each case finishes.
- In local `standalone` mode, attempt to remove `bench_file.parent / "extra-info"` in the runtime per-case loop after each case finishes.
- If `extra-info/` does not exist, do nothing.
- If a filesystem entry named `extra-info` exists but is not a directory, leave it untouched.
- Run the cleanup in per-case `finally` paths so it happens after both successful and failed cases.

## Verification

- Add unit coverage that confirms local `msprof` benchmark runs delete regenerated `extra-info/` directories after each case.
- Add unit coverage that confirms local `standalone` benchmark runs delete `extra-info/` between successive cases.
- Add unit coverage that confirms a non-directory `extra-info` entry is preserved.
