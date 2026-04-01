# Local `compare-perf` Subcommand

## Summary

- Add a local `compare-perf` CLI subcommand for comparing two saved perf data files.
- The command should not invoke a code agent or any skill workflow.
- Comparison should key off `latency-<id>:` entries so it remains stable even when file order changes.

## CLI shape

- Add:
  - `compare-perf --baseline <path> --compare <path>`
- Resolve both paths locally and fail early if either file is missing.

## Comparison behavior

- Parse both perf files as `latency-<id>: <float>` records.
- Fail if either file is malformed, empty, or contains duplicate latency ids.
- Fail if the two files do not contain the same latency ids.
- When ids match, print one comparison line per id with:
  - baseline value
  - compare value
  - percentage delta relative to baseline
- Return `0` for a successful comparison run and `1` for malformed data or mismatched ids.

## Scope

- Extend the command enum and CLI parser.
- Add the local comparison helper beside the existing local benchmark runner logic.
- Update README, AGENTS, and tests for the new command.
