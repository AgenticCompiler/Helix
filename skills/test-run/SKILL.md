---
name: test-run
description: Run or validate correctness tests for a Triton or Triton Ascend operator. Use when Codex needs to execute a standalone or differential test, inspect failures, classify likely causes, or work from an operator path plus optional mode and callable hints.
---

# Test Run

Run an existing correctness test and explain the outcome in a debugging-friendly way.

Use this skill when the user wants to execute a generated test, use either `standalone` or `differential` execution, or understand why a test failed.

## Inputs

- Operator path
- Optional explicit test path
- Optional requested test mode
- Optional callable name hint

## Outputs

- Pass or fail result
- Concise diagnosis of likely failure category
- Suggested next action when the test fails
- In `differential` mode, the final archived comparison artifacts under `differential_results/`
- A concise summary report with test verdict and failure classification

## Required Execution Contract

- Always execute tests through a direct bash command that runs the generated Python test file.
- For `standalone` mode, follow [test-standalone-run-spec.md](references/test-standalone-run-spec.md).
- For `differential` mode, follow [test-differential-run-spec.md](references/test-differential-run-spec.md).
- When `differential` mode is used for an optimized operator, compare the archived oracle and compare result files with [compare_differential_results.py](scripts/compare_differential_results.py).
- Treat those run specs as the primary execution contract.

## Input Semantics

- Requested `standalone` mode means executing a direct correctness test.
- Requested `differential` mode means executing an oracle comparison flow.
- A callable name hint should be used when the file has multiple candidates.
- The operator path identifies the implementation that must be validated.

## Generated File Invocation

The generated test file requires `--operator-file <path>` and `--api-name <name>` arguments. The runner must always construct the bash command with these arguments, passing the operator file path and API function name to the test script.

## Workflow

1. Resolve the operator and the expected test artifact.
2. Honor the requested test mode.
3. Read the corresponding run spec and construct the bash command with `--operator-file` and `--api-name` arguments.
4. Execute the test through bash.
5. In `differential` mode, treat `TEST_RESULT.pt` as a temporary result emitted beside the test file, then ensure the final archived result is stored under `differential_results/`.
6. When the target is an optimized operator in `differential` mode, compare `differential_results/oracle_result_<operator-api-name>.pt` and `differential_results/compare_result_<optimized-stem>.pt` with the helper comparison script.
7. Report success, failure, timeout, or setup error.
8. When failing, classify whether the issue looks like environment, compiler, numerical, import, or test logic trouble.

## Quality Rules

- Keep the result easy to scan.
- Preserve the exact mode the user requested.
- Distinguish between operator bugs and test bugs when evidence exists.
- Prefer `python3 <case-file>.py` style execution unless the environment clearly requires a different interpreter path.
- Do not route through project-specific test orchestration when a direct file run satisfies the spec.

See [contracts.md](references/contracts.md) for reporting guidance, and enforce the mode-specific run spec first.
