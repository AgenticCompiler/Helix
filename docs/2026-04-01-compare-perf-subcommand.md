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

- Parse the baseline perf file as strict `latency-<id>: <float>` records.
- Parse the compare perf file by extracting only the latency ids required by the baseline.
- Ignore extra compare-side fields such as `mean_ms` or free-form summary entries.
- Fail if the baseline file is malformed, empty, or contains duplicate latency ids.
- Fail if the compare file is missing any baseline latency id or provides an invalid value for one.
- When ids match, print one comparison line per id with:
  - baseline value
  - compare value
  - percentage delta relative to baseline
- Return `0` for a successful comparison run and `1` for malformed data or missing required ids.

## Scope

- Extend the command enum and CLI parser.
- Add the local comparison helper beside the existing local benchmark runner logic.
- Update README, AGENTS, and tests for the new command.
