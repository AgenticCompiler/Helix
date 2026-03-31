# Standalone Test Run Spec

## Execution rule

- Run the generated standalone test file directly with bash.
- Preferred form:

```bash
python3 test_<op>.py
```

- Run from the directory containing the test file when relative imports or neighboring files matter.

## Artifact expectations

- The test file should be directly executable.
- Success is indicated by exit code `0`.
- If the file follows the generation spec, it should print `All tests passed!` only on success.

## What to report

- Exact command used
- Working directory if relevant
- Exit code
- Key stdout or stderr when the run fails
