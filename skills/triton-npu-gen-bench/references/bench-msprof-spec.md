## Benchmark file specification for Triton kernels

This document describes the specification for benchmark files (e.g. `bench_abs.py`) for Triton operators. The goal of the benchmark is **benchmarking and profiling only**—no correctness checking.

An operator may contain:
- kernel functions
- a public entrypoint implemented as a Triton wrapper function
- a public entrypoint implemented as a PyTorch function
- a public entrypoint implemented as a `torch.nn.Module` class

The benchmark must call the resolved public entrypoint to run the operator. Raw `@triton.jit` kernels are not valid direct harness APIs.

### 1. File naming and location

- For an operator implemented in `<op>.py` (e.g. `abs.py`), the benchmark file **must** be named **`bench_<op>.py`** in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/bench_abs.py`.

### 2. Command-line interface and main flow

The benchmark module must support two usages when run as `python -m bench_<op>` (from the directory containing `bench_<op>.py`):

The benchmark file must include this metadata header near the top of the file:

```python
# bench-mode: msprof
# api-name: <name>
# api-kind: <triton-wrapper|torch-function|torch-module>
# kernels: <name>
```

| Command | Behavior |
|--------|----------|
| `python -m bench_<op> --operator-file <operator-file> --bench <N>` | Load the resolved public entrypoint from the `<operator-file>` in the same directory (e.g. an optimized variant like `opt_abs_method1.py`) using the generated harness's `# api-name:` and `# api-kind:` contract. Then run the **N-th** benchmark case (1-based). |
| `python -m bench_<op> --num-bench` | Print the **total number** of benchmark cases for this op and exit. |

- When `--num-bench` is provided, **all other arguments MUST be optional**. The script
  **must NOT** require `--operator-file` in this mode; the following
  command must work without error:
  - `python bench_<op>.py --num-bench`
- If `--bench N` is provided, then `--operator-file` is required.
- The benchmark file must define a `main()` function that parses `--operator-file`, `--bench`, and `--num-bench`, then either prints the case count or runs the selected benchmark case.

### 3. Operator API loading

- The benchmark file **must not** rely on package imports that depend on run context, instead it must **load the operator module by file path** specified by `--operator-file`.
- If `--operator-file` is set: load `<operator-file>` from that directory, and then from the loaded module, use the generated harness's `# api-name:` and `# api-kind:` contract to obtain the operator API.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module.
- If that named API does not exist in the runtime operator file, fail explicitly instead of guessing.
- If the public entrypoint is valid but the Triton kernel names cannot be resolved safely, fail explicitly because msprof mode still requires stable `# kernels:` metadata.

#### 3.1 `triton-wrapper`

Use this kind when the public API is a Python wrapper function around Triton kernels.

```python
def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, api_name)
```

#### 3.2 `torch-function`

Use this kind when the public API is a plain PyTorch-facing function or operator entrypoint that may internally call Triton kernels.

```python
def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, api_name)
```

#### 3.3 `torch-module`

Use this kind when the public API is a `torch.nn.Module` class that can be instantiated without constructor arguments.

```python
def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entrypoint_cls = getattr(module, api_name)
    try:
        return entrypoint_cls()
    except TypeError as exc:
        raise RuntimeError(
            "torch-module entrypoints must support no-argument construction; "
            "constructor arguments are not supported in generated harnesses"
        ) from exc
```

If a `torch-module` entrypoint requires constructor arguments, fail explicitly with an actionable error about unsupported constructor arguments.

### 4. Benchmark case data (no external input)

- The benchmark file **must not** read from external files (e.g. CSV or Excel). Instead, it **must** write all benchmark case data in `bench_<op>.py` as a constant list (or similar in-file structure).
- Each case is defined by the dimensions/dtypes needed for that op (e.g. for a unary op: one dtype and one shape; for a binary op: dtype and one or more shapes, as appropriate).
- You may use multiple dtypes, but the total number of benchmark cases (shapes × dtypes)
  must be **<= 20**. For example:
  - If you use 2 dtypes, test at most 10 shapes in total.
  - If you use 3 dtypes, choose shapes so that `len(shapes) * len(dtypes) <= 20`.
- When the operator's shape space is broad enough, prefer **8-20 representative cases** instead of tiny suites.
- The generated case list should cover small, medium, and large representative shapes unless the operator's valid inputs are genuinely narrow.
- Example for a unary op: a list of `(dtype, shape)` tuples, where `shape` is a tuple of integers (e.g. `(1073741824,)` or `(64, 64)`).
- Name the list clearly (e.g. `ABS_BENCH_CASES` for abs). The number of elements is the number of benchmarks; `--num-bench` prints `len(<this_list>)`.

### 5. Parameterized benchmark dispatch

- The benchmark file **must** have a **single core function** (e.g. `run_bench(operator_api, dtype, shape)`) that takes the resolved operator callable and the case parameters (dtype, shape(s)).
- **Do not** define one function per case (e.g. no `bench_1` ... `bench_K`). In `main()`, use the `--bench N` argument (1-based) to select the N-th case from the embedded list and call `run_bench(kernel_fn, *CASES[N - 1])` directly.
- Each embedded case should carry a stable case id so normalized benchmark output can be compared by id rather than only by positional order.

### 6. Core benchmark logic (e.g. `run_bench`)

- **Execution device:** The harness **must** exercise the operator on **Ascend NPU only**. Do **not** generate benchmarks intended to run primarily on CUDA, CPU, or other accelerators, or that branch to a non-NPU device for the code under test.
- Create the required tensor(s) for the kernel with the case’s dtype and shape(s).
- **All tensors must be created on device `"npu"`** (not `"cuda"`).
- Randomized input generation is allowed, but the harness must explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.
- Do not depend on ambient global RNG state or prior random draws to keep cases stable.
- Declare fixed execution counts near the top of the file:
  - `MSPROF_WARMUP_ITERS = 5`
  - `MSPROF_REPEAT_ITERS = 50`
- **Warmup:** run the kernel **5 times** before measured repetition begins.
- **Repeat:** after warmup, run the kernel **50 times** so the profiler observes a more stable sample.
- **Do not** perform any correctness check (no comparison to reference implementation).
- **Do not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark code.

### 7. Example

```python
# bench-mode: msprof
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>

import argparse
import importlib.util
import torch

API_NAME = "<resolved_entrypoint>"
MSPROF_WARMUP_ITERS = 5
MSPROF_REPEAT_ITERS = 50
CASES = [...]

def make_inputs(case):
    torch.manual_seed(case["seed"])
    ...

def run_bench(operator_api, case):
    inputs = make_inputs(case)
    for _ in range(MSPROF_WARMUP_ITERS):
        operator_api(*inputs)
    for _ in range(MSPROF_REPEAT_ITERS):
        operator_api(*inputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file")
    parser.add_argument("--bench", type=int)
    parser.add_argument("--num-bench", action="store_true")
    args = parser.parse_args()

    if args.num_bench:
        print(len(CASES))
        return

    if args.operator_file is None or args.bench is None:
        raise SystemExit("--operator-file and --bench are required unless --num-bench is used")
    if args.bench < 1 or args.bench > len(CASES):
        raise SystemExit(f"--bench must be between 1 and {len(CASES)}")

    operator_api = load_operator_api(args.operator_file, API_NAME)
    run_bench(operator_api, CASES[args.bench - 1])


if __name__ == "__main__":
    main()
```
