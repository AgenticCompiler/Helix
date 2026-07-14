# Optimize Batch Immediate Failure Output Design

## Summary

`optimize-batch` should report each workspace failure as soon as that workspace finishes, instead of making users wait for all concurrent workspaces to finish.

## User-visible behavior

- When a workspace fails during batch optimization, write and flush `[FAIL] <workspace>: <reason>` immediately.
- Continue running the remaining workspaces as before.
- Preserve the existing final, sorted batch summary and its exit-code behavior. The failed workspace therefore appears both in the immediate notification and in the final summary.
- Keep live failure notifications readable alongside `--show-output` output by sharing its terminal-write lock.

## Implementation

- In the batch completion loop, emit a failure notification for discovery, request-validation, agent, unexpected-exception, and post-optimize-command failures when each result is known.
- Do not change scheduling, status-file updates, success output, or final result rendering.

## Verification

- Add a regression test with one failed workspace and one blocked workspace. It verifies that the failure line is observed before the blocked workspace is allowed to finish.
- Run the focused optimize runtime tests, lint, type checking, and the repository test suite.
