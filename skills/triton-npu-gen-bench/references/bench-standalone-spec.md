## Benchmark file specification for Triton Ascend NPU operators

This document defines the **standalone** benchmark contract for generated `bench_<op>.py` files.

The goal of a standalone benchmark file is to declare benchmark cases for one resolved public operator entrypoint. The benchmark runner owns execution, profiling, and perf artifact generation. The benchmark file itself is **import-only** and is **not** a directly runnable CLI script.

An operator may expose its public entrypoint as:

- a Triton wrapper function
- a PyTorch-facing function
- a `torch.nn.Module` class

The generated benchmark must target the resolved public entrypoint rather than raw `@triton.jit` kernels.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the benchmark file must be named `bench_<op>.py`.
- The benchmark file must live beside the operator file it was generated for.

### 2. Required metadata header

The file must begin with a short metadata header:

```python
# bench-mode: standalone
# api-name: <resolved_entrypoint>
# api-kind: <triton-wrapper|torch-function|torch-module>
# kernels: <resolved_kernel_names>
```

Notes:

- `# api-name:` records the resolved public entrypoint.
- `# api-kind:` records how that entrypoint should be interpreted.
- `# kernels:` records one or more Triton kernel names that the runner should aggregate from profiler output.

### 3. Standalone export contract

The module must export exactly two standalone hooks:

```python
def build_operator_api(operator_module):
    ...

def build_standalone_bench_cases(operator_api):
    ...
```

Rules:

- `build_operator_api(operator_module)` is required.
- `build_standalone_bench_cases(operator_api)` is required.
- The module must be **import-only**. Do not generate `main()`, `argparse`, or direct runtime CLI handling for standalone benchmarks.
- Do not require the runner to pass `--operator-file` into the benchmark module itself.

### 4. `build_operator_api(operator_module)`

This hook constructs the final runtime object that standalone cases will close over.

Requirements:

- The hook receives the dynamically loaded runtime operator module.
- Return the final object that benchmark cases should use.
- For `triton-wrapper`, this is usually the resolved wrapper function.
- For `torch-function`, this is usually the resolved function entrypoint.
- For `torch-module`, this should usually construct and return the final callable module object.

`torch-module` notes:

- Prefer the resolved `torch.nn.Module` class when it is the real public entrypoint.
- If constructor arguments are required and cannot be resolved safely, fail explicitly instead of guessing constructor arguments.
- Generated standalone benchmarks must still mention `torch-module` behavior clearly when relevant.

### 5. `build_standalone_bench_cases(operator_api)`

This hook returns the declared benchmark cases after `operator_api` has been built.

Each case must be a mapping with:

- `id`: required stable string case id
- `fn`: required zero-argument callable that runs one prepared case
- `warmup`: optional non-negative integer
- `repeats`: optional positive integer

Important behavior:

- `fn` should close over any prepared tensors, attrs, or module state it needs.
- `fn` should execute only the benchmarked operator body.
- Input preparation should happen outside the returned `fn` whenever practical so the profiler focuses on operator execution.
- The case id should be descriptive and stable because downstream perf artifacts use `latency-<case-id>`.

### 6. Device and benchmark assumptions

- The benchmarked operator must run on **Ascend NPU**.
- All generated benchmark inputs should target device `"npu"` for the code under test.
- Do not generate correctness checks in standalone benchmark files.
- Do not put profiler logic, perf text rendering, or timing fallback logic inside the benchmark file.
- Do not directly print latency lines from the benchmark file. The runner owns artifact generation.

### 7. Runner-owned execution model

The standalone benchmark file is consumed by `triton-npu-run-eval`:

- `run-bench` imports the benchmark file and operator file
- the runner calls `build_operator_api(operator_module)`
- the runner calls `build_standalone_bench_cases(operator_api)`
- the runner profiles each case with `torch_npu.profiler`
- the runner writes the perf artifact in the shared `msprof`-aligned text format

The standalone file therefore must not assume it will be executed as `python3 bench_<op>.py ...`.

### 8. Example shape

```python
# bench-mode: standalone
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>

import torch


def build_operator_api(operator_module):
    if "<resolved_api_kind>" == "torch-module":
        return operator_module.<resolved_entrypoint>().npu().eval()
    return getattr(operator_module, "<resolved_entrypoint>")


def build_standalone_bench_cases(operator_api):
    lhs = torch.randn((1024, 1024), dtype=torch.float16, device="npu")
    rhs = torch.randn((1024, 1024), dtype=torch.float16, device="npu")

    def case_fp16_1024():
        operator_api(lhs, rhs)

    return [
        {
            "id": "fp16_1024",
            "fn": case_fp16_1024,
            "warmup": 5,
            "repeats": 50,
        }
    ]
```
