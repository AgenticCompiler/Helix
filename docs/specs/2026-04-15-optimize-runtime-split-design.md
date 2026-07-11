# Optimize Runtime Split Design

## Summary

- Split `run_optimize_request()` into smaller helpers so the top-level function owns shared setup and cleanup while supervised and unsupervised optimize flows live in separate functions.
- Preserve optimize behavior, prompts, guidance staging, cleanup, and exit-code semantics.

## Goals

- Reduce the size and branching complexity of `run_optimize_request()`.
- Make the supervised and unsupervised optimize lifecycles readable in isolation.
- Keep the `helix -> skills` dependency direction unchanged.

## Non-Goals

- Do not change optimize prompt construction, resume behavior, or supervision semantics.
- Do not change `OptimizeRunLoop`, `SupervisedOptimizeAdapter`, or `RecoveryRunnerAdapter` behavior.
- Do not refactor unrelated optimize helpers in this change.

## Design

- Keep `run_optimize_request()` responsible for:
  - preparing staged skills
  - creating the backend runner
  - creating the guidance manager
  - performing final staged-skill cleanup
- Add `_run_supervised_optimize_request(...)` for the supervised guidance lifecycle.
- Add `_run_unsupervised_optimize_request(...)` for the unsupervised guidance lifecycle.
- Have `run_optimize_request()` dispatch to one of those helpers based on `request.supervise`.

## Verification

- Run `uv run python -m unittest tests.test_optimize_runtime -v`
- Run `uv run --group dev ruff check`
- Run `uv run pyright`
