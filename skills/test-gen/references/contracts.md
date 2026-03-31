# Test Generation Contract

## Expected file shape

- Import the operator under test explicitly.
- Create deterministic sample inputs.
- Execute the operator at least once.
- Assert correctness with a clear tolerance policy.
- Allow direct execution with `if __name__ == "__main__":` when practical.

The full authoritative requirements live in:

- [test-standalone-spec.md](test-standalone-spec.md)
- [test-differential-spec.md](test-differential-spec.md)

## Standalone mode

- Assert expected values directly.
- Prefer simple reference math inside the test when the formula is small and obvious.

## Differential mode

- Compare target output against a trusted oracle.
- Explain the oracle source when it is not obvious.
- Save any large intermediate artifacts only if the user asked for them.

## Naming guidance

- Default output names can follow `test_<operator>.py` for standalone mode.
- Differential tests can follow `differential_test_<operator>.py`.
