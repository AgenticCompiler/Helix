## Benchmark file specification for Triton Ascend NPU operators

This document describes the specification for benchmark files (e.g. `bench_abs.py`) for Triton operators. The goal of the benchmark is **benchmarking and profiling only**—no correctness checking.

An operator contains of:
- kernel functions
- an operator API function which wraps the kernel functions

The benchmark must call the operator API function to run the operator.

### 1. File naming and location

- For an operator implemented in `<op>.py` (e.g. `abs.py`), the benchmark file **must** be named **`bench_<op>.py`** in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/bench_abs.py`.

### 2. Command-line interface

The benchmark file must accept the following arguments when run as `python bench_<op>.py`:

| Argument | Required | Behavior |
|----------|----------|----------|
| `--operator-file <path>` | yes | Path to the operator source file to benchmark (e.g. `abs.py` or `opt_abs.py`). |
| `--api-name <name>` | yes | Name of the operator API function to import from that file. |

Example invocations:

```bash
# Benchmark the original operator
python3 bench_abs.py --operator-file abs.py --api-name abs_

# Benchmark an optimized variant with the same bench file
python3 bench_abs.py --operator-file opt_abs.py --api-name abs_
```

### 3. Operator API loading

- The benchmark file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module; then `getattr(module, "<api-name>")` to get the callable.

### 4. Core benchmark logic (`run_bench`)

- Create the required tensor(s) for the operator with the case's dtype and shape(s), and call the operator API function loaded via `--operator-file` and `--api-name`.
- **All tensors must be created on device `"npu"`** (not `"cuda"`).
- **Do not** perform any correctness check (no comparison to reference implementation).
- **Do not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark code.
- Iterate over all benchmark cases in the file and measure each one.
- The benchmark file **must** contain an entry point that parses `--operator-file` and `--api-name`, loads the operator, and runs all cases:
  ```python
  if __name__ == "__main__":
      main()
  ```
- Use `triton.testing.do_bench_npu` to measure performance.
- Print each case result using: `print(f"latency: {latency}")`

### 5. Warmup and active policy

- Estimate the runtime of the benchmarked callable.
- If estimated runtime is less than 10ms:
  - `warmup=1000`, `active=10000`
- Otherwise:
  - `warmup=100`, `active=1000`
- The parameter meanings are:
  - `warmup`: number of warmup runs
  - `active`: number of repeated measurement runs
