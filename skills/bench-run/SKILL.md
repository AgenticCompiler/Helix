---
name: bench-run
description: Run or analyze benchmark execution for a Triton or Triton Ascend operator. Use when Codex needs to execute benchmark code, compare timing runs, or troubleshoot performance harness issues from an operator or benchmark path plus an optional benchmark mode.
---

# Bench Run

Execute or inspect benchmark runs for a single operator.

Use this skill when the user wants to run a generated benchmark, verify that the harness works, or understand benchmark failures.

## Inputs

- Operator or benchmark path
- Optional requested benchmark mode

## Outputs

- Benchmark success or failure result
- Measured metric summary when available
- A persisted performance result file under `bench_results/`
- A short diagnosis if execution fails

## Required Execution Contract

- Always execute benchmarks through direct bash commands against the generated benchmark file.
- For `standalone` mode, follow [bench-standalone-run-spec.md](references/bench-standalone-run-spec.md).
- For `msprof` mode, follow [bench-msprof-run-spec.md](references/bench-msprof-run-spec.md).
- Treat those run specs as the primary execution contract.

## Performance Artifact Contract

- Save benchmark performance data under a sibling `bench_results/` directory.
- For an original operator input such as `input.py`, write the baseline result to `bench_results/old_perf-input.txt`.
- For an optimized operator input such as `opt_input.py`, write the optimized result to `bench_results/opt_perf-input-<pattern>.txt`.
- Store one normalized latency line per benchmark case in the form `latency: <value>`.
- Also persist the raw execution log alongside the perf file when possible.

## Input Semantics

- Requested `standalone` mode means running a timing benchmark.
- Requested `msprof` mode means running a profiling-oriented benchmark.
- The input path identifies the operator or benchmark target that should be resolved.

## Workflow

1. Resolve the benchmark artifact and execution mode.
2. Read the corresponding run spec and build the minimum bash command needed for that mode.
3. Run the benchmark through bash.
4. Extract performance data from the benchmark output according to the selected mode.
5. Save normalized latency lines into the appropriate file under `bench_results/`.
6. Report success, failure, timeout, or blocked environment.
7. Summarize metrics in plain language.
8. If the benchmark fails, distinguish harness failure from operator failure where possible.

## Quality Rules

- Prefer direct `python3 bench_<op>.py ...` execution over project-specific wrappers when the benchmark file already conforms to the spec.
- Keep the reported metrics and command lines easy to inspect.
- Do not invent extra orchestration when the benchmark file already exposes the needed CLI.
- Normalize both standalone and msprof measurements into `latency: <value>` lines before saving.

See [contracts.md](references/contracts.md) for reporting guidance, and enforce the mode-specific run spec first.
