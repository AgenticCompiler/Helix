# Bench Case Support Flattening And Verbose Staging Design

## Summary

Parallel benchmark case workspaces currently preserve relative paths for every staged file. That behavior is useful for benchmark inputs and operator files, but it is unnecessary for support scripts copied into temporary case workspaces and it obscures verbose output. We will keep benchmark inputs layout-stable while flattening staged support files into the case workspace root and emitting explicit verbose staging logs.

## Goals

- Flatten staged support files such as `standalone_bench_runtime.py` into the temporary case workspace root.
- Preserve relative layout for benchmark files, operator files, and discovered JSON inputs.
- Emit human-readable `--verbose` logs for local and remote case workspace creation plus staged file copies.

## Non-Goals

- Do not change sequential local benchmark execution.
- Do not change benchmark metadata, perf file format, or case scheduling.
- Do not flatten benchmark or operator inputs that rely on relative layout.

## Decision

- Split case-workspace staging into two buckets:
  - layout-preserving inputs for benchmark/operator/json files
  - flattened support files for helper scripts
- Keep parallel standalone worker imports rooted at the case workspace itself, since flattened support files land in that root.
- Reuse the existing verbose stream and add explicit staging messages instead of relying only on low-level `ssh` / `scp` command echoing.

## Verification

- Add tests covering flattened local and remote support-file staging.
- Add tests covering local and remote verbose staging messages.
- Run the focused bench runner and remote execution unit suites plus strict `pyright` for touched skill scripts.
