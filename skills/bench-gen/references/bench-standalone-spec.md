## Benchmark file specification for Triton Ascend NPU operators

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

The benchmark file must include this metadata header near the top of the file:

```python
# bench-mode: standalone
# api-name: <name>
# api-kind: <triton-wrapper|torch-function|torch-module>
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

The benchmark file **must** contain a `main()` entry point that:
- parses `--operator-file`
- loads the operator entrypoint from the runtime operator file
- runs all benchmark cases

### 3. Operator API loading

- The benchmark file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module.
- `# api-name:` identifies the symbol to load, and `# api-kind:` identifies which loading pattern this generated harness follows.
- If that named API does not exist in the runtime operator file, fail explicitly instead of guessing.

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

### 4. Core benchmark logic (`run_bench`)

- Create the required tensor(s) for the operator with the case's dtype and shape(s), and call the resolved public entrypoint loaded via `--operator-file` and the embedded metadata.
- **All tensors must be created on device `"npu"`** (not `"cuda"`).
- **Do not** perform any correctness check (no comparison to reference implementation).
- **Do not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark code.
- Iterate over all benchmark cases in the file and measure each one.
- Assign each benchmark case a stable case id and print results using that id so downstream comparisons do not depend on stdout order alone.
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

### 6. Example

```python
# bench-mode: standalone
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernel: <resolved_kernel_name>

import argparse
import importlib.util
import triton

API_NAME = "<resolved_entrypoint>"
CASES = [("case-1", ...), ("case-2", ...)]

def make_inputs(case):
    ...

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
