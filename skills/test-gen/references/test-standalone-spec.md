## Standalone test file specification for Triton operators

This document describes standalone correctness test files for Triton Ascend operators. The goal is to produce a self-contained test script that creates deterministic NPU inputs, calls the operator API, and verifies correctness against a PyTorch reference implementation.

An operator contains:
- kernel functions
- an operator API function which wraps the kernel functions

The test must call the operator API function directly.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the test file **`test_<op>.py`** should be placed in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/test_abs.py`.

### 2. Command-line interface

The test file must accept the following arguments when run as `python test_<op>.py`:

| Argument | Required | Behavior |
|----------|----------|----------|
| `--operator-file <path>` | yes | Path to the operator source file to test (e.g. `abs.py` or `opt_abs.py`). |
| `--api-name <name>` | yes | Name of the operator API function to import from that file. |

Example invocations:

```bash
# Test the original operator
python3 test_abs.py --operator-file abs.py --api-name abs_

# Test an optimized variant with the same test file
python3 test_abs.py --operator-file opt_abs.py --api-name abs_
```

### 3. Operator API loading

- The test file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module; then `getattr(module, "<api-name>")` to get the callable.
- Do not introduce pytest or unittest scaffolding.
- The test file must remain directly executable with `python test_<op>.py --operator-file <path> --api-name <name>`.

### 4. Test case construction

- Build **multiple deterministic** test cases and call the **operator API function** with each case.
- Use **at least 3 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Set seeds where appropriate so the test data is reproducible.

### 5. Reference correctness behavior

- For each case, call the **operator API function** with the constructed inputs.
- Compute a corresponding PyTorch reference result for the same inputs.
- Compare operator output and reference output with assertions.
- Use direct assertions in the generated test function; do not save outputs to disk.
- If the operator returns tensors or tuples/lists of tensors, validate the returned values appropriately.

### 6. Test function structure

- Define a single standalone test entry function.
- The function should:
  - prepare the deterministic cases
  - run the operator on every case
  - compute the PyTorch reference result
  - assert correctness for every case
- Keep the code concise and practical.
- The full generated file should stay within roughly **140 lines**.

### 7. Entry point requirement

The file must define a `main()` function that:
1. Parses `--operator-file` and `--api-name` with `argparse`
2. Loads the operator API via `importlib`
3. Calls the test function with the loaded operator API
4. Prints `"All tests passed!"` on success

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
    print("All tests passed!")

if __name__ == "__main__":
    main()
```

- The test function (e.g. `test_xxx`) must accept the operator API callable as its first argument.
- Running the file directly should execute the standalone test and print `"All tests passed!"` only when every assertion succeeds.
