## Benchmark file specification for Triton kernels

This document describes the specification for benchmark files (e.g. `bench_abs.py`) for Triton operators. The goal of the benchmark is **benchmarking and profiling only**—no correctness checking.

An operator contains of:
- kernel functions
- an operator API function which wraps the kernel functions

The benchmark must call the operator API function to run the operator.

### 1. File naming and location

- For an operator implemented in `<op>.py` (e.g. `abs.py`), the benchmark file **must** be named **`bench_<op>.py`** in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/bench_abs.py`.

### 2. Command-line interface

The benchmark module must support two usages when run as `python -m bench_<op>` (from the directory containing `bench_<op>.py`):

| Command | Behavior |
|--------|----------|
| `python -m bench_<op> --operator-file <operator-file> --api-name <api-name> --bench <N>` | Load the **API function** from the `<operator-file>` in the same directory (e.g. an optimized variant like `opt_abs_method1.py`). Then run the **N-th** benchmark case (1-based). |
| `python -m bench_<op> --num-bench` | Print the **total number** of benchmark cases for this op and exit. |

- When `--num-bench` is provided, **all other arguments MUST be optional**. The script
  **must NOT** require `--operator-file` or `--api-name` in this mode; the following
  command must work without error:
  - `python bench_<op>.py --num-bench`
- If `--bench N` is provided, then `--operator-file` and `--api-name` are required.

### 3. Operator API loading

- The benchmark file **must not** rely on package imports that depend on run context, instead it must **load the operator module by file path** specified by `--operator-file`.
- If `--operator-file` is set: load `<operator-file>` from that directory, and then from the loaded module, use the **API function** specified by `--api-name` as the operator API. For example, if `--api-name` is `abs_`, then the benchmark always calls this same function name (e.g. `abs_`) on the loaded module.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module; then `getattr(module, "<api-name>")` to get the callable.

### 4. Benchmark case data (no external input)

- The benchmark file **must not** read from external files (e.g. CSV or Excel). Instead, it **must** write all benchmark case data in `bench_<op>.py` as a constant list (or similar in-file structure).
- Each case is defined by the dimensions/dtypes needed for that op (e.g. for a unary op: one dtype and one shape; for a binary op: dtype and one or more shapes, as appropriate).
- You may use multiple dtypes, but the total number of benchmark cases (shapes × dtypes)
  must be **≤ 10**. For example:
  - If you use 2 dtypes, test at most 5 shapes in total.
  - If you use 3 dtypes, choose shapes so that `len(shapes) * len(dtypes) <= 10`.
- Example for a unary op: a list of `(dtype, shape)` tuples, where `shape` is a tuple of integers (e.g. `(1073741824,)` or `(64, 64)`).
- Name the list clearly (e.g. `ABS_BENCH_CASES` for abs). The number of elements is the number of benchmarks; `--num-bench` prints `len(<this_list>)`.

### 5. Parameterized benchmark dispatch

- The benchmark file **must** have a **single core function** (e.g. `run_bench(operator_api, dtype, shape)`) that takes the operator api function and the case parameters (dtype, shape(s)).
- **Do not** define one function per case (e.g. no `bench_1` ... `bench_K`). In `main()`, use the `--bench N` argument (1-based) to select the N-th case from the embedded list and call `run_bench(kernel_fn, *CASES[N - 1])` directly.

### 6. Core benchmark logic (e.g. `run_bench`)

- Create the required tensor(s) for the kernel with the case’s dtype and shape(s).
- **All tensors must be created on device `"npu"`** (not `"cuda"`).
- **Warmup:** run the kernel **5 times** in a row. No need to add extra logic for msprof; the last run can be used for profiling externally.
- **Do not** perform any correctness check (no comparison to reference implementation).
- **Do not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark code.
