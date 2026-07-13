# Relative Bench Output Root Normalization

## Summary

Local benchmark profiling should treat `HELIX_BENCH_OUTPUT_DIR` as a
single configured root, even when the user passes a relative path such as
`./tmp`. Today some local benchmark paths create preserved run directories from
that relative string and later hand the resulting relative paths to isolated
case workers, which makes profiler artifacts land in temporary worker
workspaces and disappear during cleanup.

## Goals

- Keep relative `HELIX_BENCH_OUTPUT_DIR` values working for local
  `run-bench`.
- Make preserved local profiler directories stable across standalone and msprof
  benchmark flows, including parallel case execution.
- Keep the change scoped to local benchmark artifact retention.

## Non-Goals

- Do not change remote benchmark artifact handling.
- Do not change benchmark perf file naming or comparison behavior.
- Do not introduce a second environment variable or compatibility alias.

## Decision

- Normalize configured local benchmark output roots to absolute paths before any
  preserved run directory or case output directory is created.
- Apply the same normalization in the standalone runtime helper so sequential
  standalone profiling and worker subprocess handoff agree on the same resolved
  location.
- Preserve the existing error when the configured path exists and is not a
  directory.

## Verification

- Add a parallel standalone regression test that uses a relative
  `HELIX_BENCH_OUTPUT_DIR` and asserts the preserved run directory passed
  to worker subprocesses is absolute.
- Add a parallel msprof regression test that uses a relative
  `HELIX_BENCH_OUTPUT_DIR` and asserts the `--output=` directory is
  absolute.
- Run focused unit tests plus strict file-scoped `pyright` for modified skill
  scripts.
