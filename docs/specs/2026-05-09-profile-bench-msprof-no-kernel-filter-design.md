# Profile Bench Msprof No Kernel Filter Design

## Context

`profile-bench --bench-mode msprof` currently builds `msprof` commands with `--kernel-name=<name>`. The deployed `msprof` command does not support that option, so profiling fails before the benchmark case can run.

`run-bench --bench-mode msprof` still needs kernel metadata for latency aggregation. `profile-bench` is different: it collects a profiler directory for human inspection and report generation, so it should not ask `msprof` to filter by kernel.

## User-Visible Semantics

- `profile-bench --bench-mode msprof` queries `--num-bench`, selects `--bench <N>` with the existing default of `1`, and profiles that benchmark case.
- Local `profile-bench --bench-mode msprof` invokes `msprof <python> <bench-file> --operator-file <operator-file> --bench <N>`.
- Remote `profile-bench --bench-mode msprof` invokes `msprof op python3 <bench-file> --operator-file <operator-file> --bench <N>`.
- `--case-id` remains invalid for `msprof` mode.
- `--kernel-name` is no longer used to build the `msprof` command. The parser may continue accepting it temporarily for compatibility, but docs should no longer instruct users to pass it.
- Multi-kernel benchmark metadata no longer blocks `profile-bench` in `msprof` mode because kernel filtering is not part of the profiling contract.

## Implementation Notes

- Remove profile-time kernel-name resolution from `profile_runner.py`.
- Keep benchmark case validation and profile directory validation unchanged.
- Update tests to assert no `--kernel-name` argument appears in local or remote `msprof` commands.
- Update run-eval and README docs so users select a benchmark case with `--bench`, not a kernel with `--kernel-name`.
