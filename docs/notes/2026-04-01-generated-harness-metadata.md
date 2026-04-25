# Generated Harness Metadata Header

## Summary

- Generated test and benchmark harnesses should no longer require a runtime `--api-name` flag.
- The generator resolves the public entrypoint once, then records it in a small comment header near the top of the generated file.
- The generated harness remains reusable across original and optimized operator variants by accepting only `--operator-file` at runtime plus any existing benchmark-only flags.

## User-visible behavior

- Generated standalone tests run as:
  - `python3 test_<op>.py --operator-file <path>`
- Generated differential tests run as:
  - `python3 differential_test_<op>.py --operator-file <path>`
- Generated standalone benchmarks run as:
  - `python3 bench_<op>.py --operator-file <path>`
- Generated msprof benchmarks run as:
  - `python3 bench_<op>.py --num-bench`
  - `python3 bench_<op>.py --operator-file <path> --bench <N>`

## Required header metadata

- Test files must include:
  - `# test-mode: <mode>`
  - `# api-name: <resolved-entrypoint>`
  - `# api-kind: <triton-wrapper|torch-function|torch-module>`
- Benchmark files must include:
  - `# bench-mode: <mode>`
  - `# api-name: <resolved-entrypoint>`
  - `# api-kind: <triton-wrapper|torch-function|torch-module>`

These lines are intended for both human inspection and future machine parsing.

## Runtime loading contract

- The generated harness loads the operator module from `--operator-file` with `importlib`.
- The generated harness loads the callable named by its embedded `api-name` metadata.
- The generated harness uses `api-kind` to decide whether to call the symbol directly or instantiate it first.
- Runtime does not re-infer the entrypoint type from the target operator file.
- `torch-module` entrypoints are limited to no-argument construction in the generated harness contract.
- If the named API is missing from the runtime operator file, the harness must fail explicitly with an actionable error instead of guessing.

## Scope for this change

- Update generation-side skills and normative spec documents.
- Update repository examples and README guidance for generated harness usage.
- Do not redesign `run-test` or `run-bench` in this change. Their follow-up simplification can happen after the new generated-file contract is established.
