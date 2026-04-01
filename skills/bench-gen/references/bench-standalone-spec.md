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

The benchmark file must include this metadata header near the top of the file:

```python
# bench-mode: standalone
# api-name: <name>
# kernel: <name>
```

The benchmark file must accept the following arguments when run as `python bench_<op>.py`:

| Argument | Required | Behavior |
|----------|----------|----------|
| `--operator-file <path>` | yes | Path to the operator source file to benchmark (e.g. `abs.py` or `opt_abs.py`). |

Example invocations:

```bash
# Benchmark the original operator
python3 bench_abs.py --operator-file abs.py

# Benchmark an optimized variant with the same bench file
python3 bench_abs.py --operator-file opt_abs.py
```

### 3. Operator API loading

- The benchmark file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module; then `getattr(module, "<embedded-api-name>")` to get the callable.
- The generated file's `# api-name:` metadata is the source of truth for the wrapper API name.
- If that named API does not exist in the runtime operator file, fail explicitly instead of guessing.

### 4. Core benchmark logic (`run_bench`)

- Create the required tensor(s) for the operator with the case's dtype and shape(s), and call the operator API function loaded via `--operator-file` and the embedded metadata.
- **All tensors must be created on device `"npu"`** (not `"cuda"`).
- **Do not** perform any correctness check (no comparison to reference implementation).
- **Do not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark code.
- Iterate over all benchmark cases in the file and measure each one.
- Assign each benchmark case a stable case id and print results using that id so downstream comparisons do not depend on stdout order alone.
- The benchmark file **must** contain an entry point that parses `--operator-file`, loads the operator, and runs all cases:
  ```python
  # bench-mode: standalone
  # api-name: <resolved_wrapper_api>
  # kernel: <resolved_kernel_name>

  API_NAME = "<resolved_wrapper_api>"

  def load_operator_api(operator_file: str, api_name: str):
      spec = importlib.util.spec_from_file_location("operator_module", operator_file)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      return getattr(module, api_name)

  def run_bench(operator_api):
      for case_id, case in CASES:
          def bench_fn():
              return operator_api(*make_inputs(case))

          warmup, active = select_bench_config(bench_fn)
          latency = triton.backends.ascend.testing.do_bench_npu(
              bench_fn, warmup=warmup, active=active
          )
          print(f"latency-{case_id}: {latency}")

  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--operator-file", required=True)
      args = parser.parse_args()
      operator_api = load_operator_api(args.operator_file, API_NAME)
      run_bench(operator_api)

  if __name__ == "__main__":
      main()
  ```
- Use `triton.backends.ascend.testing.do_bench_npu` to measure performance.
- Print each case result using: `print(f"latency-<id>: {latency}")`

### 5. Warmup and active policy

- Estimate the runtime of the benchmarked callable.
- If estimated runtime is less than 10ms:
  - `warmup=1000`, `active=10000`
- Otherwise:
  - `warmup=100`, `active=1000`
- The parameter meanings are:
  - `warmup`: number of warmup runs
  - `active`: number of repeated measurement runs
