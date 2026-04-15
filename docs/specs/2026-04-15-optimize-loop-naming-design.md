# Optimize Loop Naming Design

## Summary

This change improves readability in the optimize orchestration path without changing runtime behavior. The current names blur three different responsibilities: optimize request entrypoints, supervised role adapters, and the loop that coordinates retries, resume, and worker/supervisor progression.

## Goals

- Make optimize loop types readable without tracing every caller.
- Clarify that the object in `supervisor.py` coordinates the run loop rather than representing the supervisor agent.
- Clarify that the supervised runner object in `orchestration.py` is an adapter over backend execution rather than the top-level orchestrator.
- Keep behavior, prompts, retries, resume flow, and file layout stable.

## Non-Goals

- No file splitting in this refactor.
- No prompt or retry semantic changes.
- No public CLI behavior changes.

## Naming Changes

- Rename `OptimizeController` to `OptimizeRunLoop`.
- Rename `SupportsSupervisedRoundRunner` to `SupportsSupervisedOptimizeAdapter`.
- Rename `SupervisedRoundRunner` to `SupervisedOptimizeAdapter`.
- Rename `RunnerWithStreams` to `RecoveryRunnerAdapter`.
- Rename orchestration helpers so their names describe entrypoint handling rather than generic "run supervised/unsupervised optimize request".
- Rename `supervisor.py` to `run_loop.py` so the file matches its loop-coordination responsibility.

## Orchestration Changes

`run_optimize_request` remains the public optimize execution entrypoint. Internally it should delegate to an `execution.py` module that owns the supervised and unsupervised execution helpers plus the backend adapter types. After this split, `orchestration.py` keeps only request construction and top-level execution entry.

## Verification

- Update targeted unit tests first so the rename is exercised through existing behavior checks.
- Run lint, pyright, and the full unittest suite after the refactor.
