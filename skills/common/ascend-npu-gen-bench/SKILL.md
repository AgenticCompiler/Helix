---
name: ascend-npu-gen-bench
description: Generate benchmark code for an Ascend NPU operator. Use when there is a need to create a new benchmark file for a given operator.
---

# Bench Gen

Generate an Ascend NPU operator benchmark script for one operator implementation.

In skill references below, `<Language>` is `triton` or `tilelang` depending on the kernel language of the current session.

## Operator File Assumption

- An operator file may contain multiple kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`).
- The operator file must expose one public entrypoint that should be benchmarked.
- The supported entrypoint kinds are:
  - `triton-wrapper`: a Python wrapper function that calls Triton kernel functions
  - `tilelang-wrapper`: a Python wrapper function that calls TileLang kernel functions
  - `torch-function`: a plain PyTorch-facing function or operator entrypoint that may internally call NPU kernels
  - `torch-module`: a `torch.nn.Module` class that represents the operator or model entrypoint and supports no-argument construction
- When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function and that wrapper launches the NPU kernel, prefer the module class as the public entrypoint rather than selecting the intermediate wrapper function.
- Benchmark generation targets the resolved public entrypoint, not the raw kernel functions.
- If no valid public entrypoint can be identified, stop and explain that benchmark generation cannot proceed safely.

## Inputs

- Operator source code or an operator file path
- The requested bench mode informs the generator about how the benchmark will be executed, but the generated file does not include a `# bench-mode:` header. Bench mode is a runtime concern.
- A requested output path should become the final destination for the benchmark.
- Auto-fix means running the generated benchmark and repairing the generated benchmark file rather than the operator when the generated harness fails.

## Outputs

- **Two** runnable Python benchmark files:
  - `bench_<op>.py` — `torch-npu-profiler` benchmark following [bench-spec.md](references/bench-spec.md)
  - `bench_<op>_msprof.py` — `msprof` benchmark following [bench-msprof-spec.md](references/bench-msprof-spec.md)
- A brief note describing benchmark assumptions and what each script measures

## Required Spec Compliance

- The generated `bench_<op>.py` must follow [bench-spec.md](references/bench-spec.md).
- The generated `bench_<op>_msprof.py` must follow [bench-msprof-spec.md](references/bench-msprof-spec.md).
- Treat each spec file as the mandatory output contract for its respective benchmark file.

## Generated File Metadata and Contract

The generated benchmark file must include a short metadata header near the top of the file:

- `# api-name: <resolved-entrypoint>`
- `# api-kind: <triton-wrapper|tilelang-wrapper|torch-function|torch-module>`
- `# kernels: <resolved_kernel_names>`

The generator must not omit `# bench-mode:`:

- In `bench_<op>.py`: set `# bench-mode: torch-npu-profiler`
- In `bench_<op>_msprof.py`: set `# bench-mode: msprof`

Across both benchmark files, the generated file must be an **import-only** module that exports:

- `build_operator_api(operator_module)`
- `build_bench_cases()`
- `build_bench_case_fn(operator_api, case)`

Across both benchmark files, keep the generated benchmark focused on the benchmark contract itself:

- do not turn the benchmark file into a self-executing command-line program
- do not add extra runtime interfaces beyond the required metadata and hooks
- let external execution tooling handle loading, selection, and profiling

Across benchmark modes, keep case data reproducible: if the generated harness uses randomized inputs, explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.

## Validation Commands

Use the ascend-npu-run-eval skill to execute generated benchmark cases.
Use `run-bench` as the standard execution command for generated benchmarks.

- For command details and required inputs, read only the focused `run-bench` guide from that skill.
- Validate the original-operator or optimized-operator path that matches the outer task.
- If the outer task is marked for remote execution, carry the same remote settings into the `run-bench` path.
- Pass the according `bench-mode` to the `run-bench` command.

## Workflow

1. Read the operator file and resolve the public entrypoint, `api-kind`, one or more NPU kernel names, and realistic benchmark inputs.
   - When the file has a `Model -> wrapper -> kernel` structure, resolve the module class as the entrypoint if it is a valid no-argument `torch-module`.
2. If the public entrypoint is missing, ambiguous, or unsafe to use, stop and report the problem instead of guessing.
3. Read the unified benchmark spec [bench-spec.md](references/bench-spec.md).
4. **Generate `bench_<op>.py`**: Create a benchmark harness with `# bench-mode: torch-npu-profiler` following the unified spec.
   - If the harness uses randomized input generation, fix the seed inside case construction.
   - Implement `build_operator_api(operator_module)`, `build_bench_cases()`, and `build_bench_case_fn(operator_api, case)`.
5. **Generate `bench_<op>_msprof.py`**: Read [bench-msprof-spec.md](references/bench-msprof-spec.md) and create a second benchmark harness with `# bench-mode: msprof`.
   - **Critical**: In `build_bench_case_fn`, all tensor creation must use `torch.rand(..., device="cpu").to("npu")` instead of `torch.rand(..., device="npu")`. Under `msprof op simulator`, direct NPU-side random tensor generation produces all-zero data.
   - This rule applies to all tensor factory functions: `torch.rand`, `torch.randn`, `torch.randint`, `torch.full`, `torch.ones`, `torch.zeros`, etc.
   - The msprof file should otherwise mirror the same cases, shapes, and entrypoint as the torch-npu-profiler file, differing only in the tensor-creation pattern.
6. If auto-fix is active, validate the generated `bench_<op>.py` with `run-bench` through the focused run-eval guide instead of doing a separate syntax-only check.
7. If validation fails, repair only the benchmark file according to the self-repair rules below, retry, and then return both runnable scripts plus a short assumptions summary.
   - For Triton Ascend compile, JIT, launch, or kernel-side failures, consult the `triton-npu-repair-guide` skill as a diagnostic reference before deciding on the smallest safe benchmark-side change.
   - This workflow still owns only the generated benchmark file. Do not treat `triton-npu-repair-guide` as permission to edit the operator file here.

## Quality Rules

- Generated benchmarks **must run on Ascend NPU** (`torch.npu`). Do **not** generate harnesses whose primary path executes the operator on CUDA, CPU, or other devices. See the unified spec for normative device rules.
- Always emit the required `# bench-mode:` metadata line in each file: `torch-npu-profiler` for `bench_<op>.py` and `msprof` for `bench_<op>_msprof.py`.
- Always generate **both** benchmark files. Do not skip the msprof variant.
- For `bench_<op>_msprof.py`: all tensor creation in `build_bench_case_fn` must use `torch.rand(..., device="cpu").to("npu")` instead of `torch.rand(..., device="npu")`. Direct NPU-side random tensor generation produces all-zero data under `msprof op simulator`.
- The msprof benchmark file should mirror the same cases and entrypoint as the torch-npu-profiler file, diverging only in the tensor-creation pattern.
- Measure the operator body, not one-time setup.
- Prefer stable repeated timing over a single run.
- Keep generated code easy to edit by hand.
- Randomized input generation is allowed only when the generated harness explicitly fixes the seed so repeated runs of the same harness produce identical inputs.
- In `torch-npu-profiler` mode, do not embed timing or profiler logic in the benchmark file. The runner owns `torch_npu.profiler` execution and perf artifact generation.
- Do not violate interface, naming, warmup, artifact, or output rules from the unified spec.
- Do not spend a separate step on syntax-only checking; rely on `run-bench` as the validation path.
- When auto-fix mode is active, only repair the generated benchmark file; do not modify the operator file.
- Do not treat raw kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`) as direct harness APIs.
- Do not guess constructor arguments for `torch-module`; if no-argument construction is not safe, stop with an explicit explanation.
- For `msprof`, fail explicitly if the public entrypoint is usable but the kernel names cannot be resolved safely.

## Self-Repair on Failure

When auto-fix mode is active and the generated benchmark fails, repair the benchmark file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

For Ascend NPU compile, JIT, launch, or kernel-side failures, you may consult the corresponding `<Language>-npu-repair-guide` skill to classify the symptom first, but any resulting edit in this workflow must stay inside the generated benchmark file. If the failure is clearly operator-side and cannot be resolved from the benchmark alone, stop and report that blocker explicitly.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or benchmark workload so the script finishes within the execution limit |
| **Compiler error** (Ascend NPU toolchain) | Use the corresponding `<Language>-npu-repair-guide` skill to help classify whether the symptom is harness-induced, then regenerate a fresh benchmark for the same operator and mode rather than patching line by line. If the failure is clearly operator-side, stop and report it. |
| **General error** (CLI, shape mismatch, runtime, etc.) | Apply a minimal targeted fix — preserve the overall benchmark structure |
| **ModuleNotFoundError** or environment issue | Report that the benchmark cannot be fixed from inside the benchmark file alone |

After any repair, always preserve the metadata header and the shared import-only hook export pattern in both files.

Always enforce the respective benchmark spec first.
