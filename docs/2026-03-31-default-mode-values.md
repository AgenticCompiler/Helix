## Summary

- `gen-test` should resolve `--test-mode` to `standalone` when the user does not pass the flag.
- `gen-bench` should resolve `--bench-mode` to `standalone` when the user does not pass the flag.
- `run-test` and `run-bench` should prefer the generated harness metadata when the user does not pass an explicit mode override.
- Prompts sent to code agents should still include explicit resolved modes for generation and optimization flows.

## User-Visible Behavior

- Generation and optimization commands always tell the agent which mode to use.
- Local run commands infer their mode from generated harness metadata by default, which keeps the CLI aligned with the file it is executing.

## Implementation Notes

- Keep parser defaults explicit for generation and optimization commands.
- Leave `run-test` and `run-bench` mode flags optional so the CLI can read the harness metadata when appropriate.
- Cover both metadata inference and prompt defaults with tests.
