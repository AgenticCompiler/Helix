# Differential Test Run Spec

## Execution rule

- Run the generated differential test file directly with bash.
- Preferred form:

```bash
python3 differential_test_<op>.py --operator-file <operator-file> --api-name <api-name>
```

- The `--operator-file` argument specifies the operator source file to test (e.g. `abs.py` or `opt_abs.py`).
- The `--api-name` argument specifies the operator API function name.
- Run from the directory containing the test file when relative paths matter.

## Artifact expectations

- Success is indicated by exit code `0`.
- A successful direct test run should first create a temporary `TEST_RESULT.pt` in the same directory as the differential test file.
- That temporary file must then be archived under `differential_results/` beside the operator file.
- The final archived filenames should follow the existing implementation contract:
  - baseline or original operator result: `differential_results/oracle_result_<operator-api-name>.pt`
  - current operator result: `differential_results/compare_result_<operator-file-stem>.pt`
- If the operator under test is the original operator file itself, the workflow may stop after ensuring the oracle artifact exists.
- If the operator under test is an optimized operator, the run is not complete until the archived oracle and compare files have been compared.
- Use the helper script in this skill to compare them:

```bash
python3 scripts/compare_differential_results.py \
  differential_results/oracle_result_<operator-api-name>.pt \
  differential_results/compare_result_<operator-file-stem>.pt
```

- Lack of `TEST_RESULT.pt` after a zero exit code should be treated as a failed or non-compliant run.
- Lack of the final archived `.pt` file under `differential_results/` should also be treated as a failed or non-compliant run.

## What to report

- Exact command used (including `--operator-file` and `--api-name` values)
- Whether `TEST_RESULT.pt` was produced
- Whether the final archived result was stored under `differential_results/`
- Final result artifact path
- Differential compare command and compare verdict when testing an optimized operator
- Exit code
- Key stdout or stderr when the run fails

## Summary report

After the run completes, produce a concise summary including:

- Operator file and API name tested
- Test mode (differential)
- Number of test cases executed
- Whether `TEST_RESULT.pt` was produced and archived
- Final verdict: pass or fail
- For optimized operators: oracle-vs-compare comparison result and tolerance level used
- If failed: failure classification and suspected root cause
