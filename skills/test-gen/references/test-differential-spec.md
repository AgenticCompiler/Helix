## Differential test file specification for Triton operators

This document describes differential comparison test files for Triton Ascend operators. The goal is to produce a self-contained test script that creates deterministic NPU inputs, calls the operator API, and saves ordered outputs for downstream comparison.

An operator contains:
- kernel functions
- an operator API function which wraps the kernel functions

The differential test must call the operator API function directly and save outputs for later comparison.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the test file **`differential_test_<op>.py`** should be in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/differential_test_abs.py`.

### 2. Imports and operator loading

- The operator API will be imported during running so you don't need to import it in the test file.
- Do not introduce pytest or unittest scaffolding.
- The test file must remain directly executable with `python differential_test_<op>.py`.

### 3. Test case construction

- Build **multiple deterministic** test cases with different shapes and dtypes and call the operator API function with each case.
- Use **at least 3 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Set seeds so the test data is reproducible.

### 4. Differential output behavior

- For each case, call the operator API function with the constructed inputs.
- Do **not** perform result assertions in this file.
- Collect outputs into a `results` list.
- Each entry in `results` must be the operator output for one case, appended in execution order.
- Save the final ordered output list in a payload dict to `TEST_RESULT.pt` in the same directory as the test file:
  - `torch.save({"results": results}, Path(__file__).parent / "TEST_RESULT.pt")`

### 5. Test function structure

- Define a single differential test entry function.
- The function should:
  - prepare the deterministic cases
  - run the operator on every case
  - collect ordered outputs
  - save the results payload to disk under the `results` key
- Keep the code concise and practical.
- The full generated file should stay within roughly **140 lines**.

### 6. Entry point requirement

Include a direct executable entry point in the test file.

```python
if __name__ == "__main__":
    test_xxx()
```

Running the file directly should execute the differential test and save the payload file.
