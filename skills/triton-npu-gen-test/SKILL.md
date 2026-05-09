---
name: triton-npu-gen-test
description: Generate correctness test code for a Triton or Triton Ascend operator from source code and task context. Use when you need to author a new operator test file, choose between standalone and differential test styles, infer the callable under test, or honor a requested output location.
---

# Test Gen

Generate a Python correctness test for a single operator implementation.

Use this skill when the user wants a new correctness test file, wants a specific test style such as `standalone` or `differential`, or provides an explicit output destination for the generated test.

## Operator File Assumption

- An operator file may contain multiple `@triton.jit` kernel functions.
- The operator file must expose one public entrypoint that should be tested.
- The supported entrypoint kinds are:
  - `triton-wrapper`: a Python wrapper function that calls Triton kernel functions
  - `torch-function`: a plain PyTorch-facing function or operator entrypoint that may internally call Triton kernels
  - `torch-module`: a `torch.nn.Module` class that represents the operator or model entrypoint and supports no-argument construction
- When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function and that wrapper launches the Triton kernel, prefer the module class as the public entrypoint rather than selecting the intermediate wrapper function.
- Test generation targets the resolved public entrypoint, not the raw kernel functions.
- If no valid public entrypoint can be identified, stop and explain that test generation cannot proceed safely.

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

## Generated File Metadata And Runtime Contract

The generated test file must include a short metadata header near the top of the file:

- `# test-mode: <mode>`
- `# api-name: <resolved-entrypoint>`
- `# api-kind: <triton-wrapper|torch-function|torch-module>`
- `# kernels: <resolved_kernel_names>`

- Always follow the selected spec file exactly. The spec is authoritative for the mode-specific runtime shape, hook surface, artifact layout, and validation behavior.
- Keep the shared contract consistent across modes: use the metadata header, resolve the public entrypoint explicitly, generate deterministic NPU coverage, and avoid inventing extra runtime behavior that the spec does not require.

## Validation Commands

Use the `triton-npu-run-eval` skill to validate generated tests.

- Standalone: run `run-test` with `--test-mode standalone`.
- Differential: run `run-test` with `--test-mode differential`, then run `compare-result` on the archived payload when you need to compare against an oracle or optimized result.
- If validation is remote, carry the same remote flags through both commands.

## Workflow

1. Read the operator code, resolve the supported public entrypoint, and read the selected spec.
2. Generate the test file to match the selected spec exactly.
3. Validate with `run-test`; if the task is differential, follow with `compare-result` when comparison is needed.
4. If validation fails, repair the test and repeat.

## Quality Rules

- Run on Ascend NPU only.
- Follow the selected spec exactly.
- Keep the metadata header and resolve the public entrypoint explicitly.
- Use `importlib` or import-only hooks only when the spec requires them.
- Keep coverage deterministic and realistic.
- Do not guess constructor arguments for `torch-module`.
- Do not treat raw `@triton.jit` kernels as direct harness APIs.

## Self-Repair on Failure

When the generated test fails, repair the test file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or workload size so the test finishes faster |
| **Compiler error** (Triton Ascend toolchain) | Regenerate a fresh test for the same operator and mode rather than patching line by line |
| **General error** (assertion, shape mismatch, etc.) | Apply a minimal targeted fix — preserve the overall test structure |
| **ModuleNotFoundError** or environment issue | Report that the test cannot be fixed from inside the test file alone |

After any repair, always preserve the metadata header and keep the generated file compliant with the selected spec.

## Failure Handling

- If the operator signature is ambiguous, explain the ambiguity and choose the narrowest safe assumption.
- If kernel functions exist but no supported public entrypoint can be identified, stop and explain that the operator API is missing.
- If the best candidate entrypoint is a `torch-module` that requires constructor arguments, stop and explain that no-argument instantiation is required for this generation contract.
- If there is no obvious oracle for differential mode, say so and fall back to a documented reference implementation or a clearly labeled placeholder.
