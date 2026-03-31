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

- Operator source code or an operator file path
- Optional explicit callable name when the wrapper API cannot be inferred safely
- Optional requested output path for the generated test
- Optional requested test style such as `standalone` or `differential`
- Optional permission to overwrite an existing generated file
- Optional request to run the generated test and repair the generated test file if it fails

## Outputs

- A complete Python test file
- A short note describing assumptions, generated coverage, and unresolved gaps

## Required Spec Compliance

- For `standalone` mode, the generated file must follow [test-standalone-spec.md](references/test-standalone-spec.md).
- For `differential` mode, the generated file must follow [test-differential-spec.md](references/test-differential-spec.md).
- Treat those spec files as normative output requirements, not loose examples.

## Input Semantics

- Requested `standalone` mode means generating an assertion-driven self-contained test that imports the operator and checks correctness directly.
- Requested `differential` mode means generating a comparison test against an oracle or reference implementation.
- An explicit callable name should override uncertain inference.
- A requested output path should become the final destination for the generated test.
- Overwrite permission allows replacing an existing generated test artifact.
- Auto-fix mode means running the generated test and repairing the generated test file rather than the operator when the generated test fails.

## Workflow

1. Read the operator code and identify the public callable, tensor arguments, scalar arguments, shapes, dtypes, and kernel launch requirements.
2. Confirm that the file contains a wrapper API that should be tested.
3. If no wrapper API can be resolved, stop and report the problem instead of guessing.
4. Infer whether the request is better served by `standalone` or `differential` mode. If the user explicitly requested a mode, honor it.
5. Read the corresponding spec file before generating the test.
6. Generate realistic test data, shape coverage, and edge cases that match the operator signature while staying within the selected spec.
7. Prefer deterministic seeds and stable tolerance handling.
8. If the output file already exists, overwrite it only when explicit overwrite permission was given.
9. If auto-fix mode is active, run the generated test with `test-run`.
10. If that generated test fails, use `test-fix` to modify the generated test file itself, then re-run the test.
11. Produce a runnable Python file, then summarize what was assumed.

## Quality Rules

- Keep the test executable as a normal Python script.
- Prefer explicit imports over dynamic loading tricks.
- Include at least one representative happy-path case.
- Add edge cases only when they are justified by the operator contract.
- Do not invent unavailable dependencies without saying so.
- Do not violate naming, entrypoint, artifact, or output rules from the selected spec.
- When auto-fix mode is active, only repair the generated test file; do not modify the operator file.

## Failure Handling

- If the operator signature is ambiguous, explain the ambiguity and choose the narrowest safe assumption.
- If kernel functions exist but no wrapper API can be identified, stop and explain that the operator API is missing.
- If there is no obvious oracle for differential mode, say so and fall back to a documented reference implementation or a clearly labeled placeholder.

See [contracts.md](references/contracts.md) for quick guidance, and always enforce the mode-specific spec file first.
