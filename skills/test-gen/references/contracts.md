# Test Generation Contract

## CLI interface

All generated test files must accept `--operator-file` and `--api-name` CLI arguments and use `importlib` to dynamically load the operator. See the spec files for the full entry point pattern and code example.

## Authoritative specs

- [test-standalone-spec.md](test-standalone-spec.md)
- [test-differential-spec.md](test-differential-spec.md)

## Standalone mode

- Assert expected values directly.
- Prefer simple reference math inside the test when the formula is small and obvious.

## Differential mode

- Save ordered outputs for downstream comparison.
- Explain the oracle source when it is not obvious.

## Naming guidance

- Standalone: `test_<operator>.py`
- Differential: `differential_test_<operator>.py`
