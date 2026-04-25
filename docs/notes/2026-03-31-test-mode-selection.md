# Test Mode Selection

## Summary

Add explicit test-mode selection for `gen-test` and `run-test`.

## User-Visible Behavior

- `gen-test` accepts `--test-mode standalone|differential`.
- `run-test` accepts `--test-mode standalone|differential`.
- The selected mode is passed through to the code agent so it can generate or run the requested style of test.
- Commands outside the test workflow should not expose this option.

## Implementation Notes

- Store the requested test mode in the agent request object.
- Add the selected mode to prompt construction for test commands.
- Keep the option CLI-scoped to `gen-test` and `run-test` so benchmark and optimize flows stay unchanged.
