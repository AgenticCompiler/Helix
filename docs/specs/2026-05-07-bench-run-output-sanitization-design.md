# Bench Run Output Sanitization Design

## Summary

`run-bench` currently exposes profiler warnings and repeats the perf path line, which adds noise without helping benchmark consumers.

## Goals

- Suppress profiler/runtime noise during successful `run-bench` execution.
- Print the perf artifact path only once.
- Keep failure output available when `run-bench` does not succeed.

## Decision

- Successful `run-bench` output should end with a single line:
  - `Perf file: <abs-path>`
- Immediately after the perf line, print:
  - `Hint: use \`compare-perf\` to inspect this perf artifact instead of reading it directly.`
- Do not print `Return code: ...` for `run-bench` success.
- Do not print `Saved perf file to: ...`.
- If the benchmark fails, print the captured execution output so the failure remains diagnosable.
- Apply the same output contract to both the main CLI handler and the skill-local `run-command.py` entrypoint.
- Suppress live streaming output from local and remote benchmark runners by routing it to a quiet sink instead of the terminal.
- Suppress standalone profiler chatter during successful local profiling runs by redirecting profiler-side stdout/stderr away from the terminal.

## Verification

- Add tests that confirm successful `run-bench` output contains only the perf path line.
- Add tests that confirm profiler warnings are not echoed during successful standalone and msprof benchmark runs.
