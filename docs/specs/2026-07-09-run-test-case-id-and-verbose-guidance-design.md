# Run-Test Case-Id And Verbose Guidance Design

## Summary

Add optional `--case-id` support to `run-test` so agents can rerun one differential test case while repairing an operator, and update skill guidance to recommend `run-test --verbose` by default when debugging correctness failures.

## Problem

- `run-bench`, `profile-bench`, and simulator flows already expose case selection, but `run-test` always executes every differential case.
- During operator repair, agents often need to iterate on one failing case quickly instead of paying the cost of rerunning the full differential suite.
- Current skill guidance does not consistently steer agents toward `--verbose`, so they can miss the per-case failure context that `run-test` already emits.

## Goals

- Let both the top-level `triton-agent run-test` command and the staged `ascend-npu-run-eval` helper commands accept `--case-id <id>`.
- Keep default `run-test` behavior unchanged when `--case-id` is omitted.
- Make `--ref-operator-file` auto-runs honor the same selected case so one-case repair loops stay comparable.
- Update the run-eval and repair-oriented skill docs to recommend `--verbose` for repair/debug reruns.

## Non-Goals

- Do not require `--case-id` for ordinary `run-test` usage.
- Do not redesign standalone test execution.
- Do not change comparison semantics for user-supplied `--ref-result` files beyond documenting that the compared payloads should cover the same cases.

## Design

### CLI surface

- Add optional `--case-id` to:
  - `triton-agent run-test`
  - `skills/common/ascend-npu-run-eval/scripts/cli.py` run-test subcommands
- Accept the flag for all run-test entrypoints so the public CLI and staged helper stay aligned.
- Reject `--case-id` in standalone mode because standalone tests do not expose a stable selectable case list.

### Differential execution

- Extend the skill-side test runner with a small case-selection helper over the already-normalized `DifferentialTestCase` list.
- When `--case-id` is present:
  - run only the matching differential case
  - archive only that case in the saved result payload
  - raise a focused error if the case id is unknown, listing available ids
- When omitted, keep the current all-cases behavior.

### Reference-result auto-generation

- Thread `case_id` through the `--ref-operator-file` resolution path in both the top-level CLI and the staged helper.
- If `run-test` auto-generates the reference result from a reference operator, it must run the same filtered case selection as the candidate run so `compare-result` sees matching payload scope.

### Skill guidance

- Update the focused run-test guide to document:
  - `--case-id <id>` is available for differential mode
  - `--verbose` is the recommended default when diagnosing failures
- Update repair/optimization guidance that tells agents to rerun `run-test` so it explicitly says to prefer `--verbose` during repair loops.

## Verification

- Parser tests for both the top-level CLI and staged helper CLI.
- Execution-handler tests proving `case_id` is forwarded through run-test and reference-operator auto-runs.
- Test-runner tests proving differential filtering, standalone rejection, and unknown-id errors.
- Contract tests locking the updated skill wording.
