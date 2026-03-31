## Differential test file specification for Triton operators

This document describes differential comparison test files for Triton Ascend operators. The goal is to produce a self-contained test script that creates deterministic NPU inputs, calls the operator API, and saves ordered outputs for downstream comparison.

An operator contains:
- kernel functions
- an operator API function which wraps the kernel functions

The differential test must call the operator API function directly and save outputs for later comparison.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the test file **`differential_test_<op>.py`** should be in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/differential_test_abs.py`.

### 2. Command-line interface

The test file must accept the following arguments when run as `python differential_test_<op>.py`:

| Argument | Required | Behavior |
|----------|----------|----------|
| `--operator-file <path>` | yes | Path to the operator source file to test (e.g. `abs.py` or `opt_abs.py`). |
| `--api-name <name>` | yes | Name of the operator API function to import from that file. |

Example invocations:

```bash
# Test the original operator
python3 differential_test_abs.py --operator-file abs.py --api-name abs_

# Test an optimized variant with the same test file
python3 differential_test_abs.py --operator-file opt_abs.py --api-name abs_
```

### 3. Operator API loading

- The test file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module; then `getattr(module, "<api-name>")` to get the callable.
- Do not introduce pytest or unittest scaffolding.
- The test file must remain directly executable with `python differential_test_<op>.py --operator-file <path> --api-name <name>`.

### 4. Test case construction

- Build **multiple deterministic** test cases with different shapes and dtypes and call the operator API function with each case.
- Use **at least 3 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Set seeds so the test data is reproducible.

### 5. Differential output behavior

- For each case, call the operator API function with the constructed inputs.
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

### 7. Entry point requirement

The file must define a `main()` function that:
1. Parses `--operator-file` and `--api-name` with `argparse`
2. Loads the operator API via `importlib`
3. Calls the test function with the loaded operator API

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--api-name", required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("operator_module", args.operator_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    operator_api = getattr(mod, args.api_name)

    test_xxx(operator_api)

if __name__ == "__main__":
    main()
```

- The test function (e.g. `test_xxx`) must accept the operator API callable as its first argument.
- Running the file directly should execute the differential test and save the payload file.
