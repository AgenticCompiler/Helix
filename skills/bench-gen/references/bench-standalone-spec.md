## Benchmark file specification for Triton Ascend NPU operators

This document describes the specification for benchmark files (e.g. `bench_abs.py`) for Triton operators. The goal of the benchmark is **benchmarking and profiling only**—no correctness checking.

An operator contains of:
- kernel functions
- an operator API function which wraps the kernel functions

The benchmark must call the operator API function to run the operator.

### 1. File naming and location

- For an operator implemented in `<op>.py` (e.g. `abs.py`), the benchmark file **must** be named **`bench_<op>.py`** in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/bench_abs.py`.

### 2. Core benchmark logic (`run_bench`)

- Create the required tensor(s) for the operator with the case’s dtype and shape(s), and call the operator API function.
- **All tensors must be created on device `"npu"`** (not `"cuda"`).
- **Do not** perform any correctness check (no comparison to reference implementation).
- **Do not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark code.
- The benchmark file **must** contain the following entry point:
  ```python
  if __name__ == "__main__":
      run_bench()
  ```
- Use `triton.testing.do_bench_npu` to measure performance.
- Print each test result using: `print(f"latency: {latency}")`

### 3. Warmup and active policy

- Estimate the runtime of the benchmarked callable.
- If estimated runtime is less than 10ms:
  - `warmup=1000`, `active=10000`
- Otherwise:
  - `warmup=100`, `active=1000`
- The parameter meanings are:
  - `warmup`: number of warmup runs
  - `active`: number of repeated measurement runs
