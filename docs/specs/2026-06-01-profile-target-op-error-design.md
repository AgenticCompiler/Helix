# Profile Target-Op Error Handling Design

## Summary

- Make `profile-report` and `profile-bench` fail with a short user-facing stderr message when `--target-op` does not match any `op_statistic` operator.
- Include the requested operator name and the available operator list in that message.
- Preserve successful JSON output formatting; do not change the current multi-line JSON contract.

## Problem

Today `profile-report` lets a missing `--target-op` propagate as an uncaught Python exception. In normal shell usage this produces a traceback. In pipeline usage, callers often redirect stderr away and then try to parse stdout as JSON. When the report generation fails before printing stdout, downstream `json.load()` sees empty input and raises a misleading `JSONDecodeError`.

`profile-bench` has the same issue after a successful profiling run: the benchmark may succeed, but the follow-up summary step can still end in an uncaught traceback when the requested target operator does not exist in `op_statistic`.

## Design

### Reporter Error

- Replace the plain target-miss `ValueError` with a user-facing error string that includes:
  - the requested `--target-op` value
  - a sorted list of available `op_statistic` operator names
- Keep the exception type compatible with existing callers by continuing to raise `ValueError`.

### CLI Handling

- `run-command.py profile-report` should catch report-generation `ValueError` and print only the formatted message to stderr, then return exit code `1`.
- `run-command.py profile-bench` should keep printing the benchmark return code and resolved profile directory, then print the same formatted summary error to stderr and return exit code `1` when report generation fails.

### Non-Goals

- Do not change successful Markdown or JSON report content.
- Do not add fuzzy matching or case-insensitive target-operator lookup in this fix.
- Do not change downstream shell snippets; the command should simply fail more clearly.
