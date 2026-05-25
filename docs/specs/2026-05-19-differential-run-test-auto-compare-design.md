# Differential Run-Test Auto-Compare Design

## Summary

In differential mode, `run-test` should optionally perform the archived-result comparison step itself so agents can finish correctness validation with one command instead of a required follow-up `compare-result` command.

## Goals

- Keep `run-test` as the public execution entrypoint for generated tests.
- Avoid fragile file-name heuristics for finding oracle payloads.
- Let differential users opt into one-command execution when they already know the oracle result path.
- Preserve standalone behavior and keep `compare-result` available for manual inspection and reruns.

## Decision

- Add optional `--oracle-result <path>` to `run-test`.
- Add optional `--compare-level strict|balanced|relaxed` to `run-test`, only valid when `--oracle-result` is also provided.
- When `run-test` resolves to `--test-mode differential`, the test execution succeeds, and `--oracle-result` is present:
  - print the archived result path
  - compare `--oracle-result` against the newly archived result
  - return the comparison exit code as the final command result
- When `--oracle-result` is not provided, keep the current archived-result output and next-step hint.
- Reject `--oracle-result` and `--compare-level` for non-differential `run-test` usage.
- Apply the same behavior to both the main CLI wrapper and `skills/triton-npu-run-eval/scripts/run-command.py`.

## Verification

- Add CLI-handler tests for successful auto-compare and compare failure exit propagation.
- Add parser and end-to-end CLI tests for the new `run-test` options.
- Add skill-helper tests so `run-command.py` stays aligned with the top-level CLI.
- Update README and skill docs to show one-command differential validation when an oracle payload is available.
