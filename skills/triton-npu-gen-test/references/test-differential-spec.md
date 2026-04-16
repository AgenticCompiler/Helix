## Differential test file specification for Triton operators

This document describes differential comparison test files for Triton Ascend operators. The goal is to produce a self-contained test script that creates deterministic NPU inputs, calls the resolved operator entrypoint, and saves ordered outputs for downstream comparison.

An operator may contain:
- kernel functions
- a public entrypoint implemented as a Triton wrapper function
- a public entrypoint implemented as a PyTorch function
- a public entrypoint implemented as a `torch.nn.Module` class

The differential test must call the resolved public entrypoint directly and save outputs for later comparison. Raw `@triton.jit` kernels are not valid direct harness APIs.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the test file **`differential_test_<op>.py`** should be in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/differential_test_abs.py`.

### 2. Command-line interface and main flow

The test file must include this metadata header near the top of the file:

```python
# test-mode: differential
# api-name: <name>
# api-kind: <triton-wrapper|torch-function|torch-module>
# kernel: <name>
```

The test file must accept the following arguments when run as `python differential_test_<op>.py`:

| Argument | Required | Behavior |
|----------|----------|----------|
| `--operator-file <path>` | yes | Path to the operator source file to test (e.g. `abs.py` or `opt_abs.py`). |

Example invocations:

```bash
# Test the original operator
python3 differential_test_abs.py --operator-file abs.py

# Test an optimized variant with the same test file
python3 differential_test_abs.py --operator-file opt_abs.py
```

The file must define a `main()` function that:
1. Parses `--operator-file` with `argparse`
2. Loads the operator API via `importlib`
3. Calls the test function with the loaded operator API

### 3. Operator API loading

- The test file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module.
- `# api-name:` identifies the symbol to load, and `# api-kind:` identifies which loading pattern this generated harness follows.
- If that named API does not exist in the runtime operator file, fail explicitly instead of guessing.
- Do not introduce pytest or unittest scaffolding.
- The test file must remain directly executable with `python differential_test_<op>.py --operator-file <path>`.

#### 3.1 `triton-wrapper`

Use this kind when the public API is a Python wrapper function around Triton kernels.

```python
def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, api_name)
```

#### 3.2 `torch-function`

Use this kind when the public API is a plain PyTorch-facing function or operator entrypoint that may internally call Triton kernels.

```python
def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, api_name)
```

#### 3.3 `torch-module`

Use this kind when the public API is a `torch.nn.Module` class that can be instantiated without constructor arguments.

```python
def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    entrypoint_cls = getattr(mod, api_name)
    try:
        return entrypoint_cls()
    except TypeError as exc:
        raise RuntimeError(
            "torch-module entrypoints must support no-argument construction; "
            "constructor arguments are not supported in generated harnesses"
        ) from exc
```

If a `torch-module` entrypoint requires constructor arguments, fail explicitly with an actionable error about unsupported constructor arguments.

### 4. Test case construction

- **Execution device:** The harness **must** exercise the operator on **Ascend NPU only**. Do **not** generate tests intended to run primarily on CUDA, CPU, or other accelerators, or that branch to a non-NPU device for the code under test.
- Build **multiple deterministic** test cases with different shapes and dtypes and call the resolved operator entrypoint with each case.
- Use **at least 3 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Set seeds so the test data is reproducible.

### 5. Differential output behavior

- For each case, call the resolved operator entrypoint with the constructed inputs.
- Do **not** perform result assertions in this file.
- Collect outputs into a `results` list.
- Each entry in `results` must be the operator output for one case, appended in execution order.
- Save the final ordered output list in a payload dict to `TEST_RESULT.pt` in the same directory as the test file:
  - `torch.save({"results": results}, Path(__file__).parent / "TEST_RESULT.pt")`

### 6. Test function structure

- Define a single differential test entry function.
- The function should:
  - prepare the deterministic cases
  - run the operator on every case
  - collect ordered outputs
  - save the results payload to disk under the `results` key
- Keep the code concise and practical.
- The full generated file should stay within roughly **140 lines**.

### 7. Example

```python
# test-mode: differential
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernel: <resolved_kernel_name>

import argparse
import importlib.util
from pathlib import Path

import torch

API_NAME = "<resolved_entrypoint>"
CASES = [...]

def make_inputs(case):
    ...

def test_xxx(operator_api):
    results = []
    for case in CASES:
        inputs = make_inputs(case)
        results.append(operator_api(*inputs))
    torch.save({"results": results}, Path(__file__).parent / "TEST_RESULT.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", required=True)
    args = parser.parse_args()

    operator_api = load_operator_api(args.operator_file, API_NAME)
    test_xxx(operator_api)

if __name__ == "__main__":
    main()
```

- The test function (e.g. `test_xxx`) must accept the operator API callable as its first argument.
- Running the file directly should execute the differential test and save the payload file.
