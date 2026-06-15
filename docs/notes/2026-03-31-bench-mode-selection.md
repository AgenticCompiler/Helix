# Bench Mode Selection

## Summary

Add explicit benchmark-mode selection for `gen-bench` and `run-bench`.

## User-Visible Behavior

- `gen-bench` accepts `--bench-mode torch-npu-profiler|msprof`.
- `run-bench` accepts `--bench-mode torch-npu-profiler|msprof`.
- The selected mode is passed through to the code agent so it can generate or run the requested style of benchmark.
- Commands outside the benchmark workflow should not expose this option.

## Implementation Notes

- Store the requested bench mode in the agent request object.
- Add the selected mode to prompt construction for benchmark commands.
- Keep the option CLI-scoped to `gen-bench` and `run-bench` so test and optimize flows stay unchanged.
