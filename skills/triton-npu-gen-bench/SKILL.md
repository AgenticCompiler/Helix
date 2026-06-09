---
name: triton-npu-gen-bench
description: Generate benchmark code for a Triton or Triton Ascend operator. Use when Codex needs to author a new benchmark file, choose between standalone and msprof styles, infer the callable under benchmark, or honor a requested output location.
---

# Bench Gen

Generate a Python benchmark script for one operator implementation.

Use this skill when the user needs a performance benchmark file for standalone timing or profiling-oriented execution.

## Operator File Assumption

- An operator file may contain multiple `@triton.jit` kernel functions.
- The operator file must expose one public entrypoint that should be benchmarked.
- The supported entrypoint kinds are:
  - `triton-wrapper`: a Python wrapper function that calls Triton kernel functions
  - `torch-function`: a plain PyTorch-facing function or operator entrypoint that may internally call Triton kernels
  - `torch-module`: a `torch.nn.Module` class that represents the operator or model entrypoint and supports no-argument construction
- When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function and that wrapper launches the Triton kernel, prefer the module class as the public entrypoint rather than selecting the intermediate wrapper function.
- Benchmark generation targets the resolved public entrypoint, not the raw kernel functions.
- If no valid public entrypoint can be identified, stop and explain that benchmark generation cannot proceed safely.

## Inputs

- Operator source code or an operator file path
- Requested `standalone` mode means generating a local timing benchmark with repeated execution.
- Requested `msprof` mode means generating a profiling-friendly benchmark intended for profiler capture.
- A requested output path should become the final destination for the benchmark.
- Auto-fix means running the generated benchmark and repairing the generated benchmark file rather than the operator when the generated harness fails.

## Outputs

- A runnable Python benchmark file
- A brief note describing benchmark assumptions and what the script measures

## Required Spec Compliance

- For `standalone` mode, the generated file must follow [bench-standalone-spec.md](references/bench-standalone-spec.md).
- For `msprof` mode, the generated file must follow [bench-msprof-spec.md](references/bench-msprof-spec.md).
- Treat those spec files as mandatory output contracts.

## Generated File Metadata and Contract

The generated benchmark file must include a short metadata header near the top of the file:

- `# bench-mode: <mode>`
- `# api-name: <resolved-entrypoint>`
- `# api-kind: <triton-wrapper|torch-function|torch-module>`
- `# kernels: <resolved_kernel_names>`

Across benchmark modes, the generated file must be an **import-only** module that exports:

- `build_operator_api(operator_module)`
- `build_bench_cases()`
- `build_bench_case_fn(operator_api, case)`

Across benchmark modes, keep the generated benchmark focused on the benchmark contract itself:

- do not turn the benchmark file into a self-executing command-line program
- do not add extra runtime interfaces beyond the required metadata and hooks
- let external execution tooling handle loading, selection, and profiling

Across benchmark modes, keep case data reproducible: if the generated harness uses randomized inputs, explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.

## Validation Commands

Use the triton-npu-run-eval skill to execute generated benchmark cases.
Use `run-bench` as the standard execution command for generated benchmarks.

- For command details and required inputs, read only the focused `run-bench` guide from that skill.
- Validate the original-operator or optimized-operator path that matches the outer task.
- If the outer task is marked for remote execution, carry the same remote settings into the `run-bench` path.

## Workflow

1. Read the operator file and resolve the public entrypoint, `api-kind`, one or more Triton kernel names, and realistic benchmark inputs.
   - When the file has a `Model -> wrapper -> kernel` structure, resolve the module class as the entrypoint if it is a valid no-argument `torch-module`.
2. If the public entrypoint is missing, ambiguous, or unsafe to use, stop and report the problem instead of guessing.
3. Select the requested benchmark mode and read the corresponding spec before generating code.
4. Generate a benchmark harness that follows the selected spec, keeps setup separate from measurement when practical, and writes the required metadata header.
   - If the harness uses randomized input generation, fix the seed inside case construction so repeated runs of the same harness produce identical inputs.
   - For both modes, implement `build_operator_api(operator_module)`, `build_bench_cases()`, and `build_bench_case_fn(operator_api, case)` instead of a directly executable timing script.
5. If auto-fix is active, validate the generated benchmark with `run-bench` through the focused run-eval guide instead of doing a separate syntax-only check.
6. If validation fails, repair only the benchmark file according to the self-repair rules below, retry, and then return a runnable script plus a short assumptions summary.
   - For Triton Ascend compile, JIT, launch, or kernel-side failures, consult the `triton-npu-repair-guide` skill as a diagnostic reference before deciding on the smallest safe benchmark-side change.
   - This workflow still owns only the generated benchmark file. Do not treat `triton-npu-repair-guide` as permission to edit the operator file here.

## Quality Rules

- Generated benchmarks **must run on Ascend NPU** (`torch.npu`). Do **not** generate harnesses whose primary path executes the operator on CUDA, CPU, or other devices. See the selected spec for normative device rules.
- Measure the operator body, not one-time setup.
- Prefer stable repeated timing over a single run.
- Keep generated code easy to edit by hand.
- Randomized input generation is allowed only when the generated harness explicitly fixes the seed so repeated runs of the same harness produce identical inputs.
- In `standalone` mode, do not embed timing or profiler logic in the benchmark file. The runner owns `torch_npu.profiler` execution and perf artifact generation.
- Do not violate interface, naming, warmup, artifact, or output rules from the selected spec.
- Do not spend a separate step on syntax-only checking; rely on `run-bench` as the validation path.
- When auto-fix mode is active, only repair the generated benchmark file; do not modify the operator file.
- Do not treat raw `@triton.jit` kernel functions as direct harness APIs.
- Do not guess constructor arguments for `torch-module`; if no-argument construction is not safe, stop with an explicit explanation.
- For `msprof`, fail explicitly if the public entrypoint is usable but the Triton kernel names cannot be resolved safely.

## Self-Repair on Failure

When auto-fix mode is active and the generated benchmark fails, repair the benchmark file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

For Triton Ascend compile, JIT, launch, or kernel-side failures, you may consult the `triton-npu-repair-guide` skill to classify the symptom first, but any resulting edit in this workflow must stay inside the generated benchmark file. If the failure is clearly operator-side and cannot be resolved from the benchmark alone, stop and report that blocker explicitly.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or benchmark workload so the script finishes within the execution limit |
| **Compiler error** (Triton Ascend toolchain) | Use `triton-npu-repair-guide` to help classify whether the symptom is harness-induced, then regenerate a fresh benchmark for the same operator and mode rather than patching line by line. If the failure is clearly operator-side, stop and report it. |
| **General error** (CLI, shape mismatch, runtime, etc.) | Apply a minimal targeted fix — preserve the overall benchmark structure |
| **ModuleNotFoundError** or environment issue | Report that the benchmark cannot be fixed from inside the benchmark file alone |

After any repair, always preserve the metadata header and the shared import-only hook export pattern.

Always enforce the mode-specific spec file first.
