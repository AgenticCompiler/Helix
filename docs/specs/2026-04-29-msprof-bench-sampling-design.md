# Msprof Benchmark Sampling Design

## Summary

The current `msprof` benchmark generation contract is too light-weight for stable profiling runs: it only warms up the operator five times, has no explicit repeat phase, and caps generated benchmark suites at ten cases. This change strengthens the `msprof` generation spec so new benchmark harnesses run a larger repeated sample and cover a broader set of representative shapes.

## Goals

- Add an explicit repeat phase to generated `msprof` benchmark harnesses.
- Increase the repeated execution count enough to make profiler statistics less sensitive to one-off noise.
- Encourage generated benchmark suites to include more representative cases instead of tiny shape lists.

## Non-Goals

- Do not change `run-bench --bench-mode msprof` runtime orchestration.
- Do not change standalone benchmark generation rules.
- Do not introduce new CLI flags for runtime warmup or repeat control.

## Decision

### Execution policy

- Keep the existing warmup phase at 5 iterations.
- Add an explicit repeat phase of 50 iterations after warmup.
- Require generated `msprof` benchmark files to declare `MSPROF_WARMUP_ITERS = 5` and `MSPROF_REPEAT_ITERS = 50` near the top of the file.

### Case coverage policy

- Raise the `msprof` benchmark case cap from 10 total cases to 20 total cases.
- Tell generators to prefer 8-20 representative cases when the operator's shape space supports it.
- Require the case list to cover small, medium, and large representative shapes rather than only a few safe defaults.

## Verification

- Tighten the generation contract test so it fails unless the `msprof` spec documents the new repeat loop, the 20-case cap, and the broader coverage guidance.
