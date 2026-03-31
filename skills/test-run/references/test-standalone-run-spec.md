# Standalone Test Run Spec

## Execution rule

- Run the generated standalone test file directly with bash.
- Preferred form:

```bash
python3 test_<op>.py --operator-file <operator-file> --api-name <api-name>
```

- The `--operator-file` argument specifies the operator source file to test (e.g. `abs.py` or `opt_abs.py`).
- The `--api-name` argument specifies the operator API function name.
- Run from the directory containing the test file when relative imports or neighboring files matter.

## Artifact expectations

- The test file should be directly executable.
- Success is indicated by exit code `0`.
- If the file follows the generation spec, it should print `All tests passed!` only on success.

## What to report

- Exact command used (including `--operator-file` and `--api-name` values)
- Working directory if relevant
- Exit code
- Key stdout or stderr when the run fails

## Summary report

After the run completes, produce a concise summary including:

- Operator file and API name tested
- Test mode (standalone)
- Number of test cases executed
- Final verdict: pass or fail
- If failed: failure classification and suspected root cause
