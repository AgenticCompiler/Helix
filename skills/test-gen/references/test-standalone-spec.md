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

The test file must include this metadata header near the top of the file:

```python
# test-mode: standalone
# api-name: <name>
# kernel: <name>
```

The test file must accept the following arguments when run as `python test_<op>.py`:

| Argument | Required | Behavior |
|----------|----------|----------|
| `--operator-file <path>` | yes | Path to the operator source file to test (e.g. `abs.py` or `opt_abs.py`). |

Example invocations:

```bash
# Test the original operator
python3 test_abs.py --operator-file abs.py

# Test an optimized variant with the same test file
python3 test_abs.py --operator-file opt_abs.py
```

### 3. Operator API loading

- The test file **must not** hard-code an import of the operator module. Instead it must **load the operator module by file path** specified by `--operator-file`.
- Use `importlib.util.spec_from_file_location` and `exec_module` to load the module; then `getattr(module, "<embedded-api-name>")` to get the callable.
- The generated file's `# api-name:` metadata is the source of truth for the wrapper API name.
- If that named API does not exist in the runtime operator file, fail explicitly instead of guessing.
- Do not introduce pytest or unittest scaffolding.
- The test file must remain directly executable with `python test_<op>.py --operator-file <path>`.

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
1. Parses `--operator-file` with `argparse`
2. Loads the operator API via `importlib`
3. Calls the test function with the loaded operator API
4. Prints `"All tests passed!"` on success

```python
# test-mode: standalone
# api-name: <resolved_wrapper_api>
# kernel: <resolved_kernel_name>

API_NAME = "<resolved_wrapper_api>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("operator_module", args.operator_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    operator_api = getattr(mod, API_NAME)

    test_xxx(operator_api)
    print("All tests passed!")

if __name__ == "__main__":
    main()
```

- The test function (e.g. `test_xxx`) must accept the operator API callable as its first argument.
- Running the file directly should execute the standalone test and print `"All tests passed!"` only when every assertion succeeds.
