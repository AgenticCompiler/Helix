# Run Commands With Explicit Operator And Artifact Paths

## Summary

Change `run-test` and `run-bench` so they no longer accept `--input`. Instead, each command must receive both:

- `--test-file` or `--bench-file`: the generated harness file to execute
- `--operator-file`: the operator implementation file to validate

This makes the run commands explicit about two different roles that are currently conflated: the executable harness file and the operator under test.

## Current State

- `run-test` and `run-bench` currently accept `--input`, treat it as the operator file, and derive the generated artifact path by convention.
- The CLI request model stores a single `input_path` and a secondary `output_path`.
- Prompt text for run commands speaks in terms of one operator input plus one requested output.
- The local `test-run` and `bench-run` skills already assume the generated file is run directly and that the command passes `--operator-file <path>` into that file.

## Problem

The current contract is too implicit for optimized-operator workflows:

- Running a generated test or benchmark often needs both the generated harness file and the operator file to execute against.
- The same generated harness may be reused against multiple operator variants, including original and optimized copies.
- Deriving the harness from the operator path hides an important input and makes the run commands less explicit.

## Goals

- Make `run-test` and `run-bench` require the exact harness file to run.
- Make the operator-under-test explicit and independent from the harness file.
- Keep generation commands unchanged.
- Keep the rest of the backend orchestration thin and skill-driven.
- Fail with short actionable CLI errors when either required path is missing.

## Non-Goals

- Do not change `gen-test`, `gen-bench`, or `optimize` input semantics in this change.
- Do not keep `--input` as a compatibility alias on `run-test` or `run-bench`.
- Do not move run-logic details from the skills into the CLI.

## Approaches

### 1. Strict explicit dual-path flags with one shared artifact flag

- `run-test --test-file <path> --operator-file <path>`
- `run-bench --test-file <path> --operator-file <path>`
- Remove `--input` entirely from these two subcommands.

Pros:

- Matches the actual execution contract of generated files.
- Makes original-versus-optimized operator validation explicit.
- Avoids hidden path derivation rules.

Cons:

- This is a breaking CLI change for the two run commands.
- Reuses test-oriented wording for a benchmark harness.

### 2. Explicit dual-path flags with command-specific artifact names

- `run-test --test-file <path> --operator-file <path>`
- `run-bench --bench-file <path> --operator-file <path>`

Pros:

- Clearest command-specific naming.
- Avoids overloading “test” for benchmark execution.

Cons:

- Slightly less uniform between the two commands.
- Diverges from the requirement as stated.

### 3. Backward-compatible transition

- Keep `--input` temporarily, add `--test-file` and `--operator-file`, and derive missing values when possible.

Pros:

- Lowest immediate migration cost.

Cons:

- Preserves ambiguity and parser complexity.
- Conflicts with the requested clean break from `--input`.

## Recommendation

Use approach 2 for this change.

Reasoning:

- It keeps the two run commands explicit without forcing benchmark users through test-oriented terminology.
- It matches the run skills better than the current derived-artifact behavior.
- It keeps the CLI explicit and predictable for optimized operator validation.

## Proposed User-Visible Behavior

### `run-test`

- Required: `--test-file <path>`
- Required: `--operator-file <path>`
- Optional: `--test-mode`, `--interact`, `--verbose`, `--show-output`, `--agent`, `--output`
- Invalid: `--input`

### `run-bench`

- Required: `--bench-file <path>`
- Required: `--operator-file <path>`
- Optional: `--bench-mode`, `--interact`, `--verbose`, `--show-output`, `--agent`, `--output`
- Invalid: `--input`

### Validation rules

- The CLI checks that both explicit harness and operator paths exist.
- The CLI exits with a short parser error when either path is missing or does not exist.
- The CLI no longer derives `test_<op>.py` or `bench_<op>.py` from the operator path for run commands.

## Proposed Internal Design

### Request model

Replace the run-command assumption that one `input_path` is enough.

Recommended model shape:

- Keep `input_path` for generation and optimize commands as the primary operator input.
- Add `operator_path` as an explicit field on `AgentRequest`.
- Interpret `output_path` for run commands as the explicit harness file path that will be executed.

Alternative acceptable shape:

- Rename `input_path` to `primary_path` and add both `operator_path` and `artifact_path`.

The first option is the smaller refactor and keeps generation commands simpler.

### Parser and command normalization

- Keep existing subcommand alias normalization.
- For `run-test`, define `--test-file` and `--operator-file` as required options.
- For `run-bench`, define `--bench-file` and `--operator-file` as required options.
- Do not register `-i` or `--input` on those two subcommands.
- Keep `--input/-i` for `gen-test`, `gen-bench`, and `optimize`.

### Path resolution

- Generation commands keep using `default_generated_output_path(...)`.
- Run commands stop calling `resolve_execution_target(...)`.
- The CLI resolves:
  - operator path from `--operator-file`
  - executed harness path from `--test-file` for `run-test`
  - executed harness path from `--bench-file` for `run-bench`
- Any remaining helper that exists only for derived run artifacts can be removed or narrowed to generation-only responsibilities.

### Working directory selection

- Use the directory containing `--test-file` as the run command workspace.

Reasoning:

- The run specs in the local skills prefer running from the directory containing the generated harness file when relative imports or sibling artifacts matter.
- Differential temporary outputs such as `TEST_RESULT.pt` are defined relative to the test file location.

### Prompt construction

Update run-command prompts so they distinguish:

- operator file under test
- harness file to execute

Recommended wording:

- `Operator input: <operator-file>`
- `Test file: <test-file>` for `run-test`
- `Benchmark file: <bench-file>` for `run-bench`

For generation commands, keep the current operator-input plus requested-output wording.

### Skill alignment

- `test-run` already expects an operator path plus optional explicit test path.
- `bench-run` already expects an operator or benchmark path and passes `--operator-file` to the benchmark script.
- The CLI should pass explicit prompt context and let the skills retain responsibility for exact bash command construction.

## Error Handling

- Missing `--test-file` on `run-test`: parser error with normal argparse usage output.
- Missing `--bench-file` on `run-bench`: parser error with normal argparse usage output.
- Missing `--operator-file`: parser error with normal argparse usage output.
- Nonexistent file path: short parser error such as `Test file path does not exist: ...`, `Bench file path does not exist: ...`, or `Operator file path does not exist: ...`.
- Existing `--output` handling remains unchanged for non-run commands.
- `run-test` and `run-bench` should no longer raise missing-derived-artifact errors, because derivation is removed.

## Testing Plan

- Parser tests for required `--test-file` and `--operator-file` on `run-test`.
- Parser tests for required `--bench-file` and `--operator-file` on `run-bench`.
- Parser tests confirming `--input` is rejected on `run-test` and `run-bench`.
- Main-path tests confirming missing file existence checks use the new error messages.
- Prompt tests confirming run prompts mention both operator file and execution file.
- Remove or rewrite tests that currently assert derived artifact resolution from an operator path.
- Verify no regression in generation and optimize command parsing.

## Documentation Updates

- Update `README.md` examples for `run-test` and `run-bench`.
- Update `AGENTS.md` to describe the new explicit run-command contract.
- Update the earlier CLI design doc so it no longer says `run-*` derives artifacts from the operator file by convention.

## Resolved Naming

- `run-test` uses `--test-file`.
- `run-bench` uses `--bench-file`.
- Both commands use `--operator-file`.
