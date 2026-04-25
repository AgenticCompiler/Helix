# Local `run-bench` Execution

## Summary

- `run-bench` should execute generated benchmark harnesses locally instead of invoking a code agent.
- Standalone mode should run the benchmark script once, capture `latency-<id>:` lines, and persist them as `<operator-filename>_perf.txt` beside the input operator file.
- Msprof mode should query the harness for case count, then run one `msprof op` command per case using the benchmark metadata header's `kernel` value.

## Standalone mode

- Execute:
  - `<python> <bench-file> --operator-file <operator-file>`
- Do not expose or require `--interact` for this command because it always streams the local process directly.
- Parse every `latency-<id>:` line from combined process output.
- Persist the normalized latency lines into:
  - `<operator-filename>_perf.txt`

## Msprof mode

- Read `# kernel: ...` from the benchmark metadata header.
- Query case count with:
  - `<python> <bench-file> --num-bench`
- For each case index, execute:
  - `msprof op --kernel-name=<kernel> <python> <bench-file> --operator-file <operator-file> --bench <N>`
- Extract `Task Duration(us): ...` and normalize it into `latency-case-<N>: <value>`.
- Persist the normalized lines into the same perf file naming scheme as standalone mode.

## Scope

- Add a local benchmark runner module.
- Route `run-bench` through the local runner path in the CLI.
- Update docs and tests for local benchmark execution behavior.
- Leave benchmark generation and optimize orchestration semantics otherwise unchanged.
