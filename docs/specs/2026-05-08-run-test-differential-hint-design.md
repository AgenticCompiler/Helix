# Run-Test Differential Hint Design

## Summary

`run-test` should give differential-test users the same kind of next-step hint that `run-bench` already gives after a successful run.

## Goals

- Print a hint only when `run-test` produces an archived differential result.
- Keep standalone `run-test` output unchanged.
- Keep the CLI entrypoint and the skill-local `run-command.py` entrypoint aligned.

## Decision

- When `run-test` succeeds in `differential` mode and returns an archived result path, print:
  - `Archived result: <path>`
  - `Hint: use \`compare-result\` to inspect this archived result instead of reading it directly.`
- Do not print the hint when no archived result exists.
- Leave the existing `run-test` return code and remote workspace output behavior unchanged.

## Verification

- Add tests that confirm successful differential `run-test` output includes the hint.
- Add tests that confirm standalone `run-test` output does not gain a new hint.
