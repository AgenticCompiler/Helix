# Local `run-test` Execution And Result Comparison

## Summary

- `run-test` should stop invoking a code agent and instead execute the generated test file directly.
- Standalone mode should report the executed process stdout, stderr, and return code.
- Differential mode should also archive the emitted result payload after a successful run.
- Add a dedicated `compare-result` subcommand for comparing two archived result payload files.

## `run-test` behavior

- Keep the existing top-level CLI shape:
  - `run-test --test-file <path> --operator-file <path> --test-mode <mode>`
- Execute the test file locally with:
  - `<python> <test-file> --operator-file <operator-file>`
- Do not build an agent prompt, launch Codex/OpenCode, or prepare skills for this command.
- Do not expose or require `--interact` for this command because it always streams the local process directly.
- Continue to print stdout, stderr, and the final return code.

## Differential archiving

- On a successful differential run, look for the generated payload file with a case-insensitive match for `TEST_RESULT.pt`.
- Search beside the input operator first because the result file is expected there for this workflow.
- Archive to the operator directory as:
  - `<operator-filename>_result.pt`
- Print the final archived path after copying succeeds.

## `compare-result` behavior

- Add `compare-result --oracle-result <path> --new-result <path>`.
- Move the differential comparison logic into the local CLI path instead of dynamically loading a helper script at runtime.
- Return the comparison helper exit code directly.

## Scope

- Implement the local execution path for `run-test`.
- Add the new `compare-result` command.
- Update user-facing docs and tests for these behaviors.
- Leave `run-bench` unchanged in this change.
