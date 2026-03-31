## Standalone test file specification for Triton operators

This document describes standalone correctness test files for Triton Ascend operators. The goal is to produce a self-contained test script that creates deterministic NPU inputs, calls the operator API, and verifies correctness against a PyTorch reference implementation.

An operator contains:
- kernel functions
- an operator API function which wraps the kernel functions

The test must call the operator API function directly.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the benchmark file **`test_<op>.py`** should be placed in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/test_abs.py`.

### 2. Imports and operator loading

- The operator API will be imported during running so you don't need to import it in the test file.
- Do not introduce pytest or unittest scaffolding.
- The test file must remain directly executable with `python test_<op>.py`.

### 3. Test case construction

- Build **multiple deterministic** test cases and call the **operator API function** with each case.
- Use **at least 3 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Set seeds where appropriate so the test data is reproducible.

### 4. Reference correctness behavior

- For each case, call the **operator API function** with the constructed inputs.
- Compute a corresponding PyTorch reference result for the same inputs.
- Compare operator output and reference output with assertions.
- Use direct assertions in the generated test function; do not save outputs to disk.
- If the operator returns tensors or tuples/lists of tensors, validate the returned values appropriately.

### 5. Test function structure

- Define a single standalone test entry function.
- The function should:
  - prepare the deterministic cases
  - run the operator on every case
  - compute the PyTorch reference result
  - assert correctness for every case
- Keep the code concise and practical.
- The full generated file should stay within roughly **140 lines**.

### 6. Entry point requirement

Include a direct executable entry point:

```python
if __name__ == "__main__":
    test_xxx()
    print("All tests passed!")
```

- Running the file directly should execute the standalone test and print `"All tests passed!"` only when every assertion succeeds.
