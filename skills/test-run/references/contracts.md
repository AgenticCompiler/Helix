# Test Run Reporting Contract

The execution contract is defined in:

- [test-standalone-run-spec.md](test-standalone-run-spec.md)
- [test-differential-run-spec.md](test-differential-run-spec.md)

## Minimum result fields

- Execution mode
- Target operator
- Target test file if known
- Final status: pass, fail, timeout, or blocked
- In `differential` mode, the final archived result path under `differential_results/`
- For optimized operators in `differential` mode, the oracle-vs-compare verdict

## Failure summary

When the test fails, prefer a short classification:

- `environment/setup`
- `import/path`
- `compiler/runtime`
- `numerical mismatch`
- `test logic`

## Follow-up guidance

- If the failure is likely environmental, say what dependency is missing.
- If the failure is likely numerical, mention tolerance and suspect inputs.
- If the failure is likely a broken generated test, say that explicitly.
