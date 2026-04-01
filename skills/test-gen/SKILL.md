---
name: test-gen
description: Generate correctness test code for a Triton or Triton Ascend operator from source code and task context. Use when Codex needs to author a new operator test file, choose between standalone and differential test styles, infer the callable under test, or honor a requested output location.
---

# Test Gen

Generate a Python correctness test for a single operator implementation.

Use this skill when the user wants a new correctness test file, wants a specific test style such as `standalone` or `differential`, or provides an explicit output destination for the generated test.

## Operator File Assumption

- An operator file may contain multiple `@triton.jit` kernel functions.
- The operator file must also contain a wrapper API function that calls those kernel functions.
- Test generation targets the wrapper API, not the raw kernel functions.
- If no valid wrapper API can be identified, stop and explain that test generation cannot proceed safely.

## Inputs

- An operator file path which contains the operator source code.
- There are two possible test styles: `standalone` and `differential`, and the user must specify one.
  - Requested `standalone` mode means generating an assertion-driven self-contained test that imports the operator and checks correctness directly.
  - Requested `differential` mode means generating a comparison test against an oracle or reference implementation.
- A requested output path should become the final destination for the generated test.

## Outputs

- A complete Python test file
- A short note describing assumptions, generated coverage, and unresolved gaps
- Naming guidance
  - Standalone: `test_<operator>.py`
  - Differential: `differential_test_<operator>.py`

## Required Spec Compliance

- For `standalone` mode, the generated file must follow [test-standalone-spec.md](references/test-standalone-spec.md).
- For `differential` mode, the generated file must follow [test-differential-spec.md](references/test-differential-spec.md).
- Treat those spec files as normative output requirements, not loose examples.

## Generated File Metadata and CLI Contract

The generated test file must include a short metadata header near the top of the file:

- `# test-mode: <mode>`
- `# api-name: <resolved-wrapper-api>`
- `# kernel: <resolved-primary-triton-kernel>`

The generated test file must accept only `--operator-file` at runtime, use `importlib` dynamic loading, and load the callable named by the embedded `api-name` metadata.

## Validation Commands

When validating a generated test, use the repository CLI subcommand `run-test` and pass both the generated test file and the operator file explicitly.

- Standalone example:
  - `python3 ../scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py --test-mode standalone`
- Differential example against the original operator:
  - `python3 ../scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file <operator>.py --test-mode differential`
- Differential example against an optimized operator:
  - `python3 ../scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --test-mode differential`

If the outer task is marked for remote execution, carry the same remote flags into validation commands.

- Remote standalone example:
  - `python3 ../scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py --remote user@host:2222`
- Remote differential example with a fixed remote root:
  - `python3 ../scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --remote user@host:2222 --remote-workdir /tmp/triton-agent`

The generated test itself is also directly runnable:

- `python3 test_<operator>.py --operator-file <operator>.py`
- `python3 differential_test_<operator>.py --operator-file opt_<operator>.py`

## Workflow

1. Read the operator code and identify the public callable, tensor arguments, scalar arguments, shapes, dtypes, and kernel launch requirements.
2. Confirm that the file contains a wrapper API that should be tested.
3. If no wrapper API can be resolved, stop and report the problem instead of guessing.
4. Read the corresponding spec file before generating the test.
5. Generate the test file according to the selected spec.
  -. Generate realistic test data, shape coverage, and edge cases that match the operator signature while staying within the selected spec.
  -  Prefer deterministic seeds and stable tolerance handling.
6. Do not add a separate syntax-check or compile-check step. Validate the generated file directly with the CLI subcommand `run-test` using one of the command patterns above.
7. If that generated test fails, infer the failure category from the raw `run-test` output and fix it; loop until the test passes.

## Quality Rules

- Keep the test executable as a normal Python script.
- Use `importlib` dynamic loading only for the operator under test via `--operator-file`. The target callable name must come from the generated file's embedded `# api-name:` metadata. All other imports should use standard explicit imports.
- Include at least one representative happy-path case.
- Add edge cases only when they are justified by the operator contract.
- Do not invent unavailable dependencies without saying so.
- Do not violate naming, entrypoint, artifact, or output rules from the selected spec.
- Do not spend a separate step on syntax-only checking; rely on `run-test` as the validation path.
- When auto-fix mode is active, only repair the generated test file; do not modify the operator file.

## Self-Repair on Failure

When the generated test fails, repair the test file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or workload size so the test finishes faster |
| **Compiler error** (Triton Ascend toolchain) | Regenerate a fresh test for the same operator and mode rather than patching line by line |
| **General error** (assertion, shape mismatch, etc.) | Apply a minimal targeted fix — preserve the overall test structure |
| **ModuleNotFoundError** or environment issue | Report that the test cannot be fixed from inside the test file alone |

After any repair, always preserve the metadata header, the `--operator-file` runtime CLI, and the `main()` entry point pattern.

## Failure Handling

- If the operator signature is ambiguous, explain the ambiguity and choose the narrowest safe assumption.
- If kernel functions exist but no wrapper API can be identified, stop and explain that the operator API is missing.
- If there is no obvious oracle for differential mode, say so and fall back to a documented reference implementation or a clearly labeled placeholder.
