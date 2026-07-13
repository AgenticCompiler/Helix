# Run-Test Case-Id And Verbose Guidance Design

## Summary

Add optional `--case-id` support to `run-test` so agents can rerun one differential test case while repairing an operator, compare that one case against an existing or regenerated reference payload without persisting `*.pt` artifacts, and update skill guidance to recommend `run-test --verbose` by default when debugging correctness failures.

## Problem

- `run-bench`, `profile-bench`, and simulator flows already expose case selection, but `run-test` always executes every differential case.
- During operator repair, agents often need to iterate on one failing case quickly instead of paying the cost of rerunning the full differential suite.
- Current skill guidance does not consistently steer agents toward `--verbose`, so they can miss the per-case failure context that `run-test` already emits.

## Goals

- Let both the top-level `helix run-test` command and the staged `ascend-npu-run-eval` helper commands accept `--case-id <id>`.
- Keep default `run-test` behavior unchanged when `--case-id` is omitted.
- Make `--ref-operator-file` auto-runs honor the same selected case so one-case repair loops stay comparable.
- Reuse an existing reference payload case when `--case-id` is paired with `--ref-result` or `--ref-operator-file` and the referenced payload already contains that case.
- Avoid persisting `*_result.pt` artifacts for both candidate and reference executions whenever `--case-id` is present.
- Update the run-eval and repair-oriented skill docs to recommend `--verbose` for repair/debug reruns.

## Non-Goals

- Do not require `--case-id` for ordinary `run-test` usage.
- Do not redesign standalone test execution.
- Do not implicitly regenerate a missing reference case when the user supplied only `--ref-result` and no reference operator is available.

## Design

### CLI surface

- Add optional `--case-id` to:
  - `helix run-test`
  - `skills/common/ascend-npu-run-eval/scripts/cli.py` run-test subcommands
- Accept the flag for all run-test entrypoints so the public CLI and staged helper stay aligned.
- Reject `--case-id` in standalone mode because standalone tests do not expose a stable selectable case list.

### Differential execution

- Extend the skill-side test runner with a small case-selection helper over the already-normalized `DifferentialTestCase` list.
- When `--case-id` is present:
  - run only the matching differential case
  - build a single-case payload in memory
  - do not persist `*_result.pt` artifacts for either the candidate or any reference rerun
  - raise a focused error if the case id is unknown, listing available ids
- When omitted, keep the current all-cases behavior.

### Single-case reference reuse

- Treat `--case-id` as an ephemeral comparison mode:
  - run the candidate case and keep the result payload in memory
  - if no reference input is provided, return the case run result without saving a `*.pt` artifact
- When `--case-id` is paired with `--ref-result`:
  - load the referenced payload
  - if it already contains the selected case, compare against that in-memory single-case payload
  - if it does not contain the selected case, fail with an actionable error because no reference operator is available to regenerate the case
- When `--case-id` is paired with `--ref-operator-file`:
  - first inspect the derived `<ref_operator>_result.pt`
  - if that payload already contains the selected case, reuse it without rerunning the reference operator
  - otherwise rerun only the selected reference case, keep that one-case payload in memory, and compare it against the candidate payload without persisting a `*.pt` artifact

### Reference-result auto-generation for full runs

- Thread `case_id` through the `--ref-operator-file` resolution path in both the top-level CLI and the staged helper.
- If `run-test` auto-generates the reference result from a reference operator in ordinary full-run mode, it must run the same filtered case selection as the candidate run so `compare-result` sees matching payload scope.

### Skill guidance

- Update the focused run-test guide to document:
  - `--case-id <id>` is available for differential mode
  - `--verbose` is the recommended default when diagnosing failures
- Update repair/optimization guidance that tells agents to rerun `run-test` so it explicitly says to prefer `--verbose` during repair loops.

## Verification

- Parser tests for both the top-level CLI and staged helper CLI.
- Execution-handler tests proving `case_id` is forwarded through run-test and reference-operator auto-runs.
- Execution-handler and staged-helper tests proving `--case-id` reuses existing reference cases, reruns missing reference cases only when `--ref-operator-file` is available, and avoids `Archived result:` output in single-case mode.
- Test-runner tests proving differential filtering, standalone rejection, unknown-id errors, single-case payload construction, and no local `*_result.pt` persistence in single-case mode.
- Contract tests locking the updated skill wording.
