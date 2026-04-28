# Msprof Bench Resilience Design

## Summary

`run-bench --bench-mode msprof` currently stops at the first failed benchmark case and returns no perf artifact for partially successful runs. It also relies only on benchmark metadata kernel declarations, which can miss newly introduced Triton kernels after optimize rounds. This change makes `msprof` benchmarking best-effort across cases, preserves mixed success and failure data in the perf artifact, and unions benchmark metadata kernels with kernels discovered from the runtime operator file.

## Goal

- Continue executing later benchmark cases after an earlier `msprof` case fails.
- Always persist a perf artifact for attempted `msprof` benchmark runs, including both successful and failed cases.
- Record actionable per-case error details in the perf artifact instead of only using plain `NA`.
- Improve kernel latency aggregation by combining benchmark metadata kernels with `@triton` kernels discovered from the runtime operator file.

## Non-Goals

- Do not change standalone benchmark behavior.
- Do not change `profile-bench --bench-mode msprof` kernel selection semantics in this work.
- Do not add a new CLI subcommand or a new top-level artifact format.
- Do not infer dynamic runtime kernel names from profiler output alone without any declared or discovered kernel source.

## Decision

### Best-effort case execution

- Local and remote `run-bench --bench-mode msprof` flows should iterate through every benchmark case from `1..N`.
- A single case failure must not stop later cases from running.
- Failures include:
  - `msprof` command exit failure
  - missing `op_statistic_*.csv`
  - malformed or empty `op_statistic` CSV content
  - any other per-case latency extraction failure after the benchmark case was launched
- After all cases finish:
  - the perf file must be written whenever at least one case was attempted
  - the command result must be non-zero if any case had an execution or parsing failure
  - `stdout` and `stderr` should remain the concatenation of all case outputs so the terminal still shows the original evidence

### Perf artifact contract

- Keep `latency-case-<N>: <value|NA>` as the comparison key line for each case.
- Continue writing `# raw-op-statistic-case-<N>: <json>` when `msprof` ran successfully enough to produce a parseable CSV.
- Add per-case error comments for actionable failures:
  - `# latency-error-case-<N>: <message>`
- Add per-case kernel resolution comments so humans can audit the aggregation source:
  - `# resolved-kernels-case-<N>: kernel_a,kernel_b,...`
  - `# kernel-source-case-<N>: metadata|operator|metadata+operator`
- When a case fails before CSV parsing succeeds:
  - write `latency-case-<N>: NA`
  - write `# latency-error-case-<N>: ...`
  - do not write `# raw-op-statistic-case-<N>` because no trustworthy payload exists
- When a case succeeds but no resolved kernel matches any `OP Type` row:
  - write `latency-case-<N>: NA`
  - write `# raw-op-statistic-case-<N>: ...`
  - write `# latency-error-case-<N>: no resolved kernels matched op_statistic csv`

### `compare-perf` behavior

- `compare-perf` should continue to ignore unrelated comment lines.
- If a case uses `latency-case-<N>: NA` with a valid `# raw-op-statistic-case-<N>` payload and no `# latency-error-case-<N>` execution failure marker beyond the "no resolved kernels matched" case, total-op fallback remains allowed.
- If a case includes `# latency-error-case-<N>` caused by execution or CSV parsing failure, `compare-perf` must fail explicitly instead of silently treating the case as a comparable `NA`.
- This preserves the existing total-op fallback for missing kernel rows while preventing broken benchmark executions from contaminating performance comparisons.

### Kernel resolution for `run-bench --bench-mode msprof`

- Resolve kernel names from two sources:
  - benchmark metadata: canonical `# kernels:` with legacy `# kernel:` fallback
  - runtime operator file: static discovery of Triton kernel definitions
- The final resolved kernel list for each run is the stable-order union of:
  - metadata kernels first
  - then any operator-discovered kernels not already listed
- Operator discovery should find Python function definitions decorated as Triton kernels, including common forms such as:
  - `@triton.jit`
  - `@triton.jit(...)`
  - decorator stacks where one decorator is `triton.jit`
- A simple AST-based detector is preferred over regex so comments and unrelated text do not create false positives.
- If metadata is present and operator discovery fails, fail explicitly for `msprof` bench mode because the union behavior is part of the measurement contract and silent fallback would hide missing optimized kernels.
- If both sources succeed but one source yields no kernels, use the non-empty source and record the corresponding `kernel-source-case-*` value.

## Error Handling

- If `--num-bench` discovery fails before any case is attempted, preserve current behavior and return no perf artifact.
- If no kernel names can be resolved from the combined metadata/operator process, fail explicitly before running cases.
- If operator kernel discovery fails because the operator file cannot be parsed as Python, fail explicitly before running cases.
- If a case-level failure happens after iteration has started, continue to later cases and write that case's `latency-error` comment.
- Temporary local and remote profiler directories must still be cleaned up for failed and successful cases, except when local artifact retention is explicitly enabled through `TRITON_AGENT_MSPROF_OUTPUT_DIR`.

## Verification

- Add local `msprof` benchmark tests that verify:
  - a failed case does not stop later cases
  - the perf file is still written with both successful and failed cases
  - failed cases emit `latency-case-* : NA` plus `# latency-error-case-*`
  - successful missing-kernel cases emit `NA` plus raw-op payload and a missing-match error comment
  - resolved kernel comments reflect metadata, operator, or combined sources
- Add remote `msprof` benchmark tests that verify the same continued-execution and persisted-artifact behavior.
- Add kernel resolution tests that verify:
  - metadata-only kernels
  - operator-only kernels
  - stable union ordering across both sources
  - duplicate removal
  - explicit failure on malformed operator source
- Add `compare-perf` coverage to verify:
  - total-op fallback still works for missing-kernel `NA`
  - comparison fails explicitly when `latency-error-case-*` marks execution or parsing failure
- Run strict file-scoped `pyright` for `skills/triton-npu-run-eval/scripts/bench_runner.py`.
