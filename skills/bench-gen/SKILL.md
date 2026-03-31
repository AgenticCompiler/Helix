---
name: bench-gen
description: Generate benchmark code for a Triton or Triton Ascend operator. Use when Codex needs to author a new benchmark file, choose between standalone and msprof styles, infer the callable under benchmark, or honor a requested output location.
---

# Bench Gen

Generate a Python benchmark script for one operator implementation.

Use this skill when the user needs a performance benchmark file for standalone timing or profiling-oriented execution.

## Operator File Assumption

- An operator file may contain multiple `@triton.jit` kernel functions.
- The operator file must also contain a wrapper API function that calls those kernel functions.
- Benchmark generation targets the wrapper API, not the raw kernel functions.
- If no valid wrapper API can be identified, stop and explain that benchmark generation cannot proceed safely.

## Inputs

- Operator source code or an operator file path
- Optional explicit callable name
- Optional requested benchmark style such as `standalone` or `msprof`
- Optional requested output path
- Optional permission to overwrite an existing generated file
- Optional request to run the generated benchmark and repair the generated benchmark file if it fails

## Outputs

- A runnable Python benchmark file
- A brief note describing benchmark assumptions and what the script measures

## Required Spec Compliance

- For `standalone` mode, the generated file must follow [bench-standalone-spec.md](references/bench-standalone-spec.md).
- For `msprof` mode, the generated file must follow [bench-msprof-spec.md](references/bench-msprof-spec.md).
- Treat those spec files as mandatory output contracts.

## Input Semantics

- Requested `standalone` mode means generating a local timing benchmark with repeated execution.
- Requested `msprof` mode means generating a profiling-friendly benchmark intended for profiler capture.
- An explicit callable name should override uncertain inference.
- A requested output path should become the final destination for the benchmark.
- Overwrite permission allows replacing an existing generated benchmark artifact.
- Auto-fix mode means running the generated benchmark and repairing the generated benchmark file rather than the operator when the generated harness fails.

## Generated File CLI Contract

The generated benchmark file must accept `--operator-file` and `--api-name` arguments with `importlib` dynamic loading, following the entry point pattern defined in the spec files. This allows the same benchmark file to be reused for both original and optimized operator variants. Msprof mode additionally requires `--bench <N>` and `--num-bench`.

## Workflow

1. Read the operator signature and infer realistic benchmark inputs.
2. Confirm that the file contains a wrapper API that should be benchmarked.
3. If no wrapper API can be resolved, stop and report the problem instead of guessing.
4. Select the benchmark style from the requested mode.
5. Read the corresponding benchmark spec before generating code.
6. Generate deterministic inputs and a clean benchmark harness that satisfies the selected spec.
7. Separate setup cost from measured execution when possible.
8. If the output file already exists, overwrite it only when explicit overwrite permission was given.
9. If auto-fix mode is active, run the generated benchmark with `bench-run`.
10. If that generated benchmark fails, infer the failure category from the raw error output and apply the matching repair strategy (see "Self-Repair on Failure" below), then re-run the benchmark.
11. Return a runnable script and a short assumptions summary.

## Quality Rules

- Measure the operator body, not one-time setup.
- Prefer stable repeated timing over a single run.
- Keep generated code easy to edit by hand.
- Do not violate CLI, naming, warmup, artifact, or output rules from the selected spec.
- When auto-fix mode is active, only repair the generated benchmark file; do not modify the operator file.

## Self-Repair on Failure

When auto-fix mode is active and the generated benchmark fails, repair the benchmark file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or benchmark workload so the script finishes within the execution limit |
| **Compiler error** (Triton Ascend toolchain) | Regenerate a fresh benchmark for the same operator and mode rather than patching line by line |
| **General error** (CLI, shape mismatch, runtime, etc.) | Apply a minimal targeted fix — preserve the overall benchmark structure |
| **ModuleNotFoundError** or environment issue | Report that the benchmark cannot be fixed from inside the benchmark file alone |

After any repair, always preserve the `--operator-file` / `--api-name` CLI interface and the `main()` entry point pattern.

See [contracts.md](references/contracts.md) for quick guidance, and always enforce the mode-specific spec file first.
