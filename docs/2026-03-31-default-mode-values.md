## Summary

- `gen-test` and `run-test` should resolve `--test-mode` to `standalone` when the user does not pass the flag.
- `gen-bench` and `run-bench` should resolve `--bench-mode` to `standalone` when the user does not pass the flag.
- Prompts sent to code agents should always include the resolved mode instead of omitting the line when the CLI uses a default.

## User-Visible Behavior

- Test commands always tell the agent which test mode to use.
- Benchmark commands always tell the agent which benchmark mode to use.
- The default mode is explicit rather than implied, so verbose prompt output matches actual CLI behavior.

## Implementation Notes

- Set parser defaults on the relevant subcommands instead of leaving the fields as `None`.
- Keep prompt rendering logic simple by always passing the resolved values from the CLI layer.
- Cover both parsing defaults and prompt text with tests.
