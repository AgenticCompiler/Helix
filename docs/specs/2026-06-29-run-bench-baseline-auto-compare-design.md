# Run-Bench Baseline Auto-Compare Design

## Goal

Add `--baseline-operator-file` to `run-bench` so the command can treat benchmark comparison as one workflow:

- reuse an existing baseline perf artifact when it already exists
- otherwise run the benchmark once for the baseline operator to create that perf artifact
- run the benchmark for the candidate operator
- automatically compare the two perf artifacts before returning

## Current Behavior

- `run-bench` accepts one benchmark file and one operator file.
- It produces one perf artifact and prints a hint telling the user to run `compare-perf` manually.
- Baseline-versus-candidate benchmarking is available as a manual multi-step workflow, not as one command.

## User-Visible Semantics

- `run-bench --baseline-operator-file <path>` becomes an optional extension of the existing command.
- When the flag is omitted, `run-bench` keeps its current behavior.
- When the flag is provided:
  - the benchmark harness still runs exactly as it does today, once per operator file
  - the command derives the baseline perf path using the same naming rule that `run_local_bench` / `run_remote_bench` already use for that baseline operator unless an explicit baseline perf artifact is already at that derived path
  - if that baseline perf artifact does not exist, `run-bench` benchmarks the baseline operator first to create it
  - the command then benchmarks the candidate operator
  - if both perf artifacts exist and both benchmark runs succeed, the command runs the existing perf comparison helper automatically and returns its exit code

## Output Contract

- Baseline-free `run-bench` output stays unchanged.
- Baseline-aware `run-bench` still prints the candidate `Perf file: <path>`.
- If the baseline run was executed in this invocation and produced a perf artifact, the command also prints `Baseline perf file: <path>`.
- On successful automatic comparison, the command prints the existing `compare-perf` output directly instead of the old hint that asked the user to run `compare-perf` manually.
- If the baseline perf file already exists and is reused, the command should not rerun the baseline benchmark and should still print `Baseline perf file: <path>`.

## Failure Semantics

- If `--baseline-operator-file` points to a missing file, argument validation fails before execution.
- If generating the baseline perf artifact fails, the command returns that failing benchmark status and does not run the candidate compare step.
- If the candidate benchmark fails, the command returns that failing benchmark status and does not run compare-perf.
- If both benchmark runs succeed but automatic perf comparison fails, the command returns the compare-perf exit code.

## Scope

- Update the top-level CLI `run-bench` parser and handler.
- Update the skill-local `skills/common/ascend-npu-run-eval/scripts/run-command.py` parser and dispatcher.
- Update the run-eval MCP server `run-bench` tool so it can express the same workflow.
- Add regression tests for parser behavior, handler behavior, and skill-local command behavior.
- Update user-facing `run-bench` documentation.

## Non-Goals

- Do not change the benchmark harness hook contract.
- Do not merge `run-bench` with `probe-bench`; they remain separate workflows.
- Do not add a new explicit baseline perf path flag in this change.
